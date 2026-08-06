import os
import sys
import cv2  
import imageio
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from collections import Counter
import random
import copy
import time
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import torch.utils.checkpoint as cp
import torchvision.models as models    
from torchvision import datasets, transforms

from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import recall_score
from scipy.stats import gmean
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import PCA 
from sklearn.model_selection import StratifiedKFold

import smote_variants as sv
from vim_model_timm import ViM_Classifier
from vit_pytorch import ViT

import gc

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAMPlusPlus

from scipy import linalg

_TB_IMPORT_ERROR = None
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as _tb_import_error:
    SummaryWriter = None
    _TB_IMPORT_ERROR = _tb_import_error

_UMAP_IMPORT_ERROR = None
try:
    import umap.umap_ as umap
except Exception as _umap_import_error:
    umap = None
    _UMAP_IMPORT_ERROR = _umap_import_error

    
# =====================================================================
# 1. Helper Functions and Loss Modules 
# =====================================================================
class AugmentedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, args, dataset_name='cifar10', train=True):
        self.data         = data
        self.labels       = labels
        self.train        = train
        self.dataset_name = dataset_name.lower()
        self.args         = args

        if self.dataset_name == 'cifar10':
            mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        elif self.dataset_name == 'cifar100':
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        elif self.dataset_name == 'dermamnist':
            mean = (0.7630, 0.5456, 0.5700)
            std  = (0.1409, 0.1526, 0.1699)
        else:
            raise ValueError(f"Unsupported dataset_name: {dataset_name}")

        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std  = torch.tensor(std).view(3, 1, 1)
        target_size = args['img_size']

        if train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((target_size, target_size)) if self.dataset_name == 'dermamnist' else transforms.Lambda(lambda x: x),
                transforms.RandomCrop(target_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip() if self.dataset_name == 'dermamnist' else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            # No augmentation: only resize and normalize
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((target_size, target_size)) if self.dataset_name == 'dermamnist' else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # GPU-RESIDENT DATA SUPPORT: self.data may live on the GPU (to
        # keep the ~20GB balanced dataset out of system RAM). torchvision's
        # ToPILImage/RandomCrop/etc. only work on CPU tensors, so pull just
        # this one sample (a few hundred KB) back to CPU before transforming
        # -- the bulk dataset never has to leave VRAM.
        x = self.data[idx].cpu()
        y = self.labels[idx].cpu()
        x = x * self.std + self.mean
        x = torch.clamp(x, 0.0, 1.0)
        x = self.transform(x)
        return x, y


def _make_summary_writer(log_dir):
    if SummaryWriter is None:
        print(f"TensorBoard unavailable ({_TB_IMPORT_ERROR}); continuing without logging.")
        class _NullWriter:
            def add_scalar(self, *args, **kwargs): return None
            def add_histogram(self, *args, **kwargs): return None
            def add_image(self, *args, **kwargs): return None
            def add_figure(self, *args, **kwargs): return None
            def add_text(self, *args, **kwargs): return None
            def close(self): return None
        return _NullWriter()
    return SummaryWriter(log_dir=log_dir)


def log_grouped_bar_chart(writer, model_name, class_names,
                          f1, recall, precision, step=0):
    x     = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, f1,        width, label='F1')
    ax.bar(x,         recall,    width, label='Recall')
    ax.bar(x + width, precision, width, label='Precision')
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title(f'Per-Class Metrics ({model_name})')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    writer.add_figure(f"PerClass/{model_name}_metrics", fig, step)
    plt.close(fig)


def transformer_reshape_transform(tensor):
    if len(tensor.shape) == 4:
        B, H, W, C = tensor.shape
        if H <= 32 and W <= 32:
            return tensor.permute(0, 3, 1, 2)
        return tensor

    if len(tensor.shape) == 3:
        import math
        B, N, C = tensor.shape
        spatial_size = int(math.sqrt(N))
        if spatial_size * spatial_size != N:
            tensor       = tensor[:, 1:, :]
            N            = N - 1
            spatial_size = int(math.sqrt(N))
        result = tensor.reshape(B, spatial_size, spatial_size, C)
        return result.permute(0, 3, 1, 2)

    return tensor


def _get_ff_layer(vit_model, block_idx):
    sublayer = vit_model.transformer.layers[block_idx][1]
    ff       = sublayer.fn if hasattr(sublayer, 'fn') else sublayer
    if hasattr(ff, 'net'):
        last_linear = None
        for m in ff.net:
            if isinstance(m, nn.Linear):
                last_linear = m
        return last_linear if last_linear is not None else ff.net[-1]
    return ff

   
# =====================================================================
# 2. The Main Pipeline Function
# =====================================================================
def derma_classification(params, args, model_names, balanced_data, balanced_labels, val_loader, test_loader):
    """
    Stage C: Takes the dataset and trains ResNet, EfficientNet, MobileNet, ViM, and ViT.

    experiment_tag controls both data handling and save paths:
      'balanced_data'             — SMOTE oversampled + augmentation
      'imbalanced_data'           — original imbalanced + augmentation
      'imbalanced_data_noaugment' — original imbalanced, NO augmentation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    dataset_name      = params['dataset_name']
    experiment_tag    = params['experiment_tag']
    classifier_epochs = args.get('classifier_epochs', 50)

    # -----------------------------------------------------------------------
    # KEY CHANGE: pass train=False for noaugment tag so AugmentedTensorDataset
    # skips RandomCrop / RandomFlip and only applies resize + normalize
    # -----------------------------------------------------------------------
    use_augmentation = (experiment_tag != 'imbalanced_data_noaugment')

    balanced_dataset = AugmentedTensorDataset(
        balanced_data,
        balanced_labels,
        args=args,
        dataset_name=dataset_name,
        train=use_augmentation      # False -> no random crop/flip
    )

    counts = {}
    for i in range(args['num_class']):
        counts[i] = (balanced_labels == i).sum().item()
    print(f'Data Distribution: {Counter(counts)}')
    print(f'Augmentation enabled: {use_augmentation}')

    print("--- Stage C: Training and Evaluating Classifiers ---")

    all_model_results = {}

    orig_val_dataset  = val_loader.dataset if hasattr(val_loader, 'dataset') else val_loader
    orig_test_dataset = test_loader.dataset if hasattr(test_loader, 'dataset') else test_loader

    # Testing NUM_WORKERS=4 (original setting) for two concurrent jobs
    # now that folds_data.pth is loaded per-fold (6GB) instead of fully
    # (30GB). This previously caused OOM when run alongside another
    # job, but that was confounded by the un-split 30GB folds_data.pth
    # also being duplicated across workers. Watching memory closely.
    NUM_WORKERS = 4
    PERSISTENT  = True if NUM_WORKERS > 0 else False

    for model_name in model_names:
        print(f"\nStarting Training for: {model_name}")

        if 'final_classifier'      in locals(): del final_classifier
        if 'classifier_optimizer'  in locals(): del classifier_optimizer
        torch.cuda.empty_cache()
        gc.collect()

        # ================== BATCH CONFIG ==================
        if model_name in ['vit', 'vim']:
            current_batch_size     = 64
            target_effective_batch = 256
        else:
            current_batch_size     = 128
            target_effective_batch = 256

        accumulation_steps = target_effective_batch // current_batch_size
        val_batch_size     = 128
        print(f"Batch={current_batch_size}, Accum={accumulation_steps}")

        # ================== DATALOADERS ==================
        balanced_train_loader = torch.utils.data.DataLoader(
            balanced_dataset,
            batch_size=current_batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=PERSISTENT,
            prefetch_factor=2 if NUM_WORKERS > 0 else None
        )

        val_loader = torch.utils.data.DataLoader(
            orig_val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=PERSISTENT
        )

        test_loader = torch.utils.data.DataLoader(
            orig_test_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=PERSISTENT
        )

        # ================== MODEL ==================
        if model_name == 'resnet18':
            final_classifier = models.resnet18(weights=None)
            final_classifier.conv1 = nn.Conv2d(args['n_channel'], 64, kernel_size=7, stride=2, padding=3, bias=False)
            final_classifier.fc    = nn.Linear(final_classifier.fc.in_features, args['num_class'])
            target_layers          = [final_classifier.layer4[-1]]

        elif model_name == 'efficientnet_b0':
            final_classifier = models.efficientnet_b0(weights=None)
            final_classifier.features[0][0]  = nn.Conv2d(args['n_channel'], 32, kernel_size=3, stride=2, padding=1, bias=False)
            final_classifier.classifier[1]   = nn.Linear(final_classifier.classifier[1].in_features, args['num_class'])
            target_layers = [final_classifier.features[-1]]

        elif model_name == 'mobilenet_v2':
            final_classifier = models.mobilenet_v2(weights=None)
            final_classifier.features[0][0] = nn.Conv2d(args['n_channel'], 32, kernel_size=3, stride=2, padding=1, bias=False)
            final_classifier.classifier[1]  = nn.Linear(final_classifier.classifier[1].in_features, args['num_class'])
            target_layers = [final_classifier.features[-1]]

        elif model_name == 'vim':
            final_classifier = ViM_Classifier(
                num_classes=args['num_class'],
                img_size=args['img_size']
            )
            backbone_obj = final_classifier.backbone if hasattr(final_classifier, 'backbone') else final_classifier
            if hasattr(backbone_obj, 'stages'):
                final_stage   = backbone_obj.stages[-1]
                target_layers = [final_stage.blocks[-1]] if hasattr(final_stage, 'blocks') and len(final_stage.blocks) > 0 else [final_stage]
            elif hasattr(backbone_obj, 'layers'):
                target_layers = [backbone_obj.layers[-1]]
            elif hasattr(backbone_obj, 'norm'):
                target_layers = [backbone_obj.norm]
            else:
                target_layers = [final_classifier.head if hasattr(final_classifier, 'head') else final_classifier]
            print(f"GradCAM Target Layer for ViM: {target_layers[0].__class__.__name__}")

        elif model_name == 'vit':
            chosen_patch_size = 8 if args['img_size'] <= 64 else (16 if args['img_size'] <= 128 else 32)
            final_classifier  = ViT(
                image_size=args['img_size'],
                patch_size=chosen_patch_size,
                num_classes=args['num_class'],
                dim=128,
                depth=6,
                heads=8,
                mlp_dim=256,
                channels=args['n_channel']
            )
            target_layers = [final_classifier.transformer.layers[-1][0]]
            print(f"GradCAM Target: last attention to_out | tokens={(args['img_size']//chosen_patch_size)**2 + 1}")

        final_classifier = final_classifier.to(device)
        final_classifier.train()

        classifier_optimizer = torch.optim.AdamW(final_classifier.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        classifier_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler            = torch.optim.lr_scheduler.ReduceLROnPlateau(classifier_optimizer, mode='min', factor=0.5, patience=5)

        # -----------------------------------------------------------------------
        # KEY CHANGE: TensorBoard log dir and model save path both include
        # experiment_tag so runs never overwrite each other
        # -----------------------------------------------------------------------
        log_dir_name  = f"{experiment_tag}_{model_name}_lr{params['lr']}_z{params['n_z']}"
        full_log_path = f"/home/s2522002/Experiment/Tensorboard_nz{params['n_z']}/{log_dir_name}"

        if os.path.exists(full_log_path):
            shutil.rmtree(full_log_path)
        writer = _make_summary_writer(full_log_path)

        best_val_f1     = 0.0
        val_losses      = []
        val_accuracies  = []
        best_model_state = None

        patience         = 30
        epochs_no_improve = 0
        min_delta        = 1e-3
        all_cam_frames   = []

        try:
            if device.type == 'cuda':
                try:
                    scaler = torch.amp.GradScaler('cuda')
                except TypeError:
                    scaler = torch.cuda.amp.GradScaler()
            else:
                scaler = torch.amp.GradScaler()
        except Exception:
            scaler = torch.amp.GradScaler()

        for epoch in range(args['classifier_epochs']):
            final_classifier.train()
            epoch_train_loss    = 0.0
            epoch_train_correct = 0
            epoch_train_total   = 0

            classifier_optimizer.zero_grad(set_to_none=True)

            for batch_idx, (imgs, labs) in enumerate(balanced_train_loader):
                imgs = imgs.to(device, non_blocking=True)
                labs = labs.to(device, non_blocking=True).view(-1).long()

                with torch.amp.autocast(device_type=device.type):
                    if model_name in ['vit', 'vim']:
                        output = cp.checkpoint(final_classifier, imgs, use_reentrant=False)
                    else:
                        output = final_classifier(imgs)
                    if isinstance(output, tuple):
                        output = output[0]
                    loss = classifier_criterion(output, labs) / accumulation_steps

                _, predicted       = torch.max(output.detach(), 1)
                epoch_train_total  += labs.size(0)
                epoch_train_correct += (predicted == labs).sum().item()
                del predicted
                del output

                scaler.scale(loss).backward()
                true_step_loss = loss.item() * accumulation_steps
                del loss

                if (batch_idx + 1) % accumulation_steps == 0 or \
                   (batch_idx + 1) == len(balanced_train_loader):
                    scaler.unscale_(classifier_optimizer)
                    torch.nn.utils.clip_grad_norm_(final_classifier.parameters(), max_norm=1.0)
                    scaler.step(classifier_optimizer)
                    scaler.update()
                    classifier_optimizer.zero_grad(set_to_none=True)

                global_step = epoch * len(balanced_train_loader) + batch_idx
                writer.add_scalar("Loss/Train_step", true_step_loss, global_step)
                epoch_train_loss += true_step_loss

            avg_train_loss = epoch_train_loss / len(balanced_train_loader)
            train_accuracy = 100.0 * epoch_train_correct / epoch_train_total

            # Validation
            final_classifier.eval()
            epoch_val_loss = 0.0
            val_preds, val_trues = [], []

            with torch.no_grad():
                for imgs, labs in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    labs = labs.to(device, non_blocking=True).view(-1).long()
                    with torch.amp.autocast(device_type=device.type):
                        output = final_classifier(imgs)
                        if isinstance(output, tuple):
                            output = output[0]
                        loss = classifier_criterion(output, labs)
                    epoch_val_loss += loss.item()
                    _, predicted = torch.max(output, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_trues.extend(labs.cpu().numpy())

            # GradCAM phase
            first_epoch = 0
            mid_epoch   = args['classifier_epochs'] // 2
            is_gradcam_epoch = (
                epoch in [first_epoch, mid_epoch]
                or (epochs_no_improve + 1 >= patience)
                or (epoch == args['classifier_epochs'] - 1)
            )

            if is_gradcam_epoch:
                if epoch == first_epoch:
                    phase_label = "Start"
                elif epochs_no_improve + 1 >= patience or epoch == args['classifier_epochs'] - 1:
                    phase_label = "End"
                else:
                    phase_label = "Mid"

                val_imgs, _ = next(iter(val_loader))
                input_img   = val_imgs[:1].to(device)

                img_show = input_img[0].detach().cpu().permute(1, 2, 0).numpy()
                img_show = (img_show - img_show.min()) / (img_show.max() - img_show.min() + 1e-8)

                if epoch == 0:
                    orig_tensor = torch.from_numpy(img_show).permute(2, 0, 1)
                    writer.add_image("GradCAM/00_Original_Image_Reference", orig_tensor, 0)

                if model_name == 'vit':
                    single_layer = final_classifier.transformer.layers[-1][0].to_out[0]
                    use_reshape  = True
                elif model_name == 'vim':
                    backbone = final_classifier.backbone if hasattr(final_classifier, 'backbone') else final_classifier
                    if hasattr(backbone, 'stages'):
                        # Use blocks[0] — visually confirmed cleaner than blocks[-1]
                        single_layer = backbone.stages[-1].blocks[0] if hasattr(backbone.stages[-1], 'blocks') else backbone.stages[-1]
                        use_reshape  = True
                    elif hasattr(backbone, 'layers'):
                        single_layer = backbone.layers[-1]
                        use_reshape  = True
                    else:
                        single_layer = final_classifier.head if hasattr(final_classifier, 'head') else final_classifier
                        use_reshape  = False
                elif model_name == 'resnet18':
                    single_layer = final_classifier.layer4[-1]
                    use_reshape  = False
                elif model_name == 'efficientnet_b0':
                    single_layer = final_classifier.features[-1]
                    use_reshape  = False
                elif model_name == 'mobilenet_v2':
                    single_layer = final_classifier.features[-1]
                    use_reshape  = False
                else:
                    single_layer = target_layers[0]
                    use_reshape  = False

                # ViT: manual GradCAM using to_out[0] and CLS gradient weighting
                if model_name == 'vit':
                    captured_act  = {}
                    captured_grad = {}

                    def forward_hook(module, inp, out):
                        captured_act['v'] = out

                    def backward_hook(module, gin, gout):
                        g = gout[0] if isinstance(gout, tuple) else gout
                        if g is not None:
                            captured_grad['v'] = g

                    fh = single_layer.register_forward_hook(forward_hook)
                    bh = single_layer.register_full_backward_hook(backward_hook)

                    try:
                        input_tensor = input_img.clone().requires_grad_(True)
                        final_classifier.eval()

                        with torch.enable_grad():
                            final_classifier.zero_grad()
                            output = final_classifier(input_tensor)
                            if isinstance(output, tuple):
                                output = output[0]
                            score = output[0, output.argmax(dim=1)[0]]
                            score.backward()

                        if 'v' in captured_act and 'v' in captured_grad:
                            act  = captured_act['v']
                            grad = captured_grad['v']

                            # CLS token (index 0) carries all gradient signal in vit_pytorch
                            cls_grad  = grad[:, 0, :].abs()           # (1, C)
                            patch_act = act[:, 1:, :]                  # (1, 49, C)
                            weights   = cls_grad.unsqueeze(1)          # (1, 1, C)
                            cam_map   = torch.relu((weights * patch_act).sum(dim=-1))  # (1, 49)

                            n_patches = args['img_size'] // chosen_patch_size
                            cam_np    = cam_map[0].detach().cpu().numpy().reshape(n_patches, n_patches)
                            cam_np    = cv2.resize(cam_np, (img_show.shape[1], img_show.shape[0]))

                            if cam_np.max() > cam_np.min():
                                cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

                            visualization = show_cam_on_image(img_show, cam_np, use_rgb=True)
                            labeled_viz   = visualization.copy()
                            cv2.putText(labeled_viz, f"Epoch: {epoch} ({phase_label})",
                                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            all_cam_frames.append(labeled_viz)
                            viz_tensor = torch.from_numpy(labeled_viz).permute(2, 0, 1)
                            writer.add_image("GradCAM/Evolution_Slider", viz_tensor, epoch)

                    except Exception as e:
                        print(f"ViT Manual GradCAM failed: {e}")
                    finally:
                        fh.remove()
                        bh.remove()
                        torch.cuda.empty_cache()

                else:
                    # CNN and ViM: pytorch_grad_cam
                    cam = GradCAMPlusPlus(
                        model=final_classifier,
                        target_layers=[single_layer],
                        reshape_transform=transformer_reshape_transform if use_reshape else None
                    )
                    try:
                        input_tensor = input_img.clone().requires_grad_(True)
                        with torch.enable_grad():
                            final_classifier.zero_grad()
                            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

                        if grayscale_cam.max() > grayscale_cam.min():
                            grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (
                                grayscale_cam.max() - grayscale_cam.min() + 1e-8)

                        visualization = show_cam_on_image(img_show, grayscale_cam, use_rgb=True)
                        labeled_viz   = visualization.copy()
                        cv2.putText(labeled_viz, f"Epoch: {epoch} ({phase_label})",
                                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        all_cam_frames.append(labeled_viz)
                        viz_tensor = torch.from_numpy(labeled_viz).permute(2, 0, 1)
                        writer.add_image("GradCAM/Evolution_Slider", viz_tensor, epoch)

                    except Exception as e:
                        print(f"GradCAM failed at epoch {epoch}: {e}")
                    finally:
                        cam.activations_and_grads.release()
                        del cam
                        torch.cuda.empty_cache()

                final_classifier.eval()

            avg_val_loss = epoch_val_loss / len(val_loader)
            val_accuracy = accuracy_score(val_trues, val_preds) * 100
            val_f1       = f1_score(val_trues, val_preds, average='macro', zero_division=0)

            if writer:
                writer.add_scalars(f"Loss_Overlay/{model_name}",      {'Train_Loss': avg_train_loss, 'Val_Loss': avg_val_loss}, epoch+1)
                writer.add_scalars(f"Accuracy_Overlay/{model_name}",  {'Train_Acc': train_accuracy,  'Val_Acc': val_accuracy},  epoch+1)
                writer.add_scalars(f"Overfitting_Check/{model_name}", {'Train_Loss': avg_train_loss, 'Val_Acc_Scaled': val_accuracy / 100.0}, epoch+1)
                writer.add_scalar("F1/Val",       val_f1,       epoch+1)
                writer.add_scalar("Loss/Train",   avg_train_loss, epoch+1)
                writer.add_scalar("Loss/Val",     avg_val_loss,   epoch+1)
                writer.add_scalar("Accuracy/Val", val_accuracy,   epoch+1)
                writer.add_scalar("LR", classifier_optimizer.param_groups[0]['lr'], epoch+1)
                writer.flush()

            scheduler.step(avg_val_loss)

            if val_f1 > best_val_f1 + min_delta:
                best_val_f1       = val_f1
                best_model_state  = copy.deepcopy(final_classifier.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

            if (epoch + 1) % 10 == 0:
                print(f"[{model_name}] Epoch {epoch+1}: Val Loss {avg_val_loss:.4f}, "
                      f"Val Acc {val_accuracy:.2f}%, Val F1 {val_f1:.4f}")

        # Save TensorBoard video
        if all_cam_frames:
            video_data   = np.stack(all_cam_frames)
            video_tensor = torch.from_numpy(video_data).permute(0, 3, 1, 2).unsqueeze(0)
            writer.add_video("GradCAM/Training_Movie", video_tensor, fps=5)

        # -----------------------------------------------------------------------
        # KEY CHANGE: save path uses experiment_tag so the three conditions
        # save to separate files and never overwrite each other:
        #   balanced_data_vit_lr0.0001_z512.pth
        #   imbalanced_data_vit_lr0.0001_z512.pth
        #   imbalanced_data_noaugment_vit_lr0.0001_z512.pth
        # -----------------------------------------------------------------------
        if best_model_state is not None:
            save_models_dir = Path(f"/home/s2522002/Experiment/Results_nz{params['n_z']}/Saved_Models")
            save_models_dir.mkdir(parents=True, exist_ok=True)

            model_weights_filename = (
                f"{experiment_tag}_{model_name.lower()}_lr{params['lr']}_z{params['n_z']}.pth"
            )
            target_save_path = save_models_dir / model_weights_filename
            torch.save(best_model_state, target_save_path)
            print(f"Saved {model_name.upper()} weights to: {target_save_path}")

        if best_model_state is not None:
            final_classifier.load_state_dict(best_model_state)
        else:
            print(f"{model_name}: no improved checkpoint — using final weights")

        # Test evaluation
        final_classifier.eval()
        all_true, all_pred, all_probs = [], [], []

        with torch.no_grad():
            for imgs, labs in test_loader:
                imgs = imgs.to(device, non_blocking=True)
                labs = labs.to(device, non_blocking=True).view(-1).long()
                output = final_classifier(imgs)
                if isinstance(output, tuple):
                    output = output[0]
                _, predicted = torch.max(output.data, 1)
                all_true.append(labs.cpu())
                all_pred.append(predicted.cpu())
                all_probs.append(output.softmax(dim=1).cpu())

        true_labels = torch.cat(all_true).numpy()
        pred_labels = torch.cat(all_pred).numpy()
        prob_labels = np.concatenate(all_probs, axis=0)

        res_f1   = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
        res_asca = balanced_accuracy_score(true_labels, pred_labels)
        res_acc  = accuracy_score(true_labels, pred_labels)
        recalls  = recall_score(true_labels, pred_labels, average=None, zero_division=0)

        precision_per_class = precision_score(true_labels, pred_labels, average=None, zero_division=0)
        recall_per_class    = recall_score(true_labels, pred_labels, average=None, zero_division=0)
        f1_per_class        = f1_score(true_labels, pred_labels, average=None, zero_division=0)
        class_names         = [f"class_{i}" for i in range(len(f1_per_class))]

        if writer:
            log_grouped_bar_chart(writer, model_name, class_names,
                                  f1_per_class, recall_per_class, precision_per_class, step=0)

            cm            = confusion_matrix(true_labels, pred_labels)
            cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

            fig, ax = plt.subplots(figsize=(7, 7))
            im      = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title(f"Normalized Confusion Matrix - {model_name}")
            fig.colorbar(im)
            tick_marks = np.arange(args['num_class'])
            ax.set_xticks(tick_marks); ax.set_xticklabels(tick_marks, rotation=45)
            ax.set_yticks(tick_marks); ax.set_yticklabels(tick_marks)
            thresh = cm_normalized.max() / 2.
            for ii, jj in itertools.product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
                ax.text(jj, ii, format(cm_normalized[ii, jj], '.2f'),
                        horizontalalignment="center",
                        color="white" if cm_normalized[ii, jj] > thresh else "black")
            ax.set_ylabel('True label')
            ax.set_xlabel('Predicted label')
            plt.tight_layout()
            writer.add_figure(f"ConfusionMatrix/{model_name}", fig, global_step=0)
            writer.flush()
            plt.close(fig)

        res_gm = gmean(recalls)
        try:
            res_auc = roc_auc_score(true_labels, prob_labels, multi_class='ovr', average='macro')
        except:
            res_auc = 0.0

        all_model_results[model_name] = {
            'f1': res_f1, 'asca': res_asca, 'acc': res_acc,
            'auc': res_auc, 'gmean': res_gm
        }

        print(f"{model_name} Final Test | ACC: {res_acc:.4f} | AUC: {res_auc:.4f} | "
              f"F1: {res_f1:.4f} | GM: {res_gm:.4f}\n")

        if writer:
            writer.close()

        print(f"Finished {model_name}. Cleaning up memory...")
        all_cam_frames.clear()
        del final_classifier
        del classifier_optimizer
        if 'scheduler' in locals():
            del scheduler
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "="*30 + " OVERALL SUMMARY " + "="*30)
    for m_name, metrics in all_model_results.items():
        print(f"{m_name.ljust(15)} | ACC: {metrics['acc']:.4f} | AUC: {metrics['auc']:.4f} | "
              f"F1: {metrics['f1']:.4f} | GM: {metrics['gmean']:.4f}")

    return all_model_results