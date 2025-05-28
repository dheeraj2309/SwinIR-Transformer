# main_test_swinir.py (with additions for training and freezing)

import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests

# --- ADDED IMPORTS for Training ---
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# --- END ADDED IMPORTS ---

from models.network_swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util


# --- ADDED CustomSRDataset Class for Training ---
class CustomSRDataset(Dataset):
    def __init__(self, lq_dir, gt_dir, scale_factor, task):
        self.lq_files = sorted(glob.glob(os.path.join(lq_dir, '*')))
        self.gt_files = sorted(glob.glob(os.path.join(gt_dir, '*')))
        self.scale_factor = scale_factor
        self.task = task # To know if it's gray_dn for single channel loading

        if not self.lq_files:
            raise FileNotFoundError(f"No LQ images found in {lq_dir}")
        if not self.gt_files:
            raise FileNotFoundError(f"No GT images found in {gt_dir}")
        if len(self.lq_files) != len(self.gt_files):
            print(f"Warning: Mismatch in number of LQ ({len(self.lq_files)}) and GT ({len(self.gt_files)}) images.")
            # You might want to implement a more robust pairing mechanism if names don't perfectly align
            # For now, we'll proceed if they have the same count, assuming correspondence by sort order.
            # If counts are different, this will likely lead to errors later.

    def __len__(self):
        return min(len(self.lq_files), len(self.gt_files)) # Use min to avoid index out of bounds if counts differ

    def __getitem__(self, idx):
        lq_path = self.lq_files[idx]
        gt_path = self.gt_files[idx] # Assumes direct correspondence by sorted order

        if self.task == 'gray_dn':
            img_lq_bgr = cv2.imread(lq_path, cv2.IMREAD_GRAYSCALE)
            img_gt_bgr = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if img_lq_bgr is None: raise ValueError(f"Could not read LQ image: {lq_path}")
            if img_gt_bgr is None: raise ValueError(f"Could not read GT image: {gt_path}")
            # Add channel dimension
            img_lq_bgr = np.expand_dims(img_lq_bgr, axis=2)
            img_gt_bgr = np.expand_dims(img_gt_bgr, axis=2)
            img_lq = img_lq_bgr.astype(np.float32) / 255.
            img_gt = img_gt_bgr.astype(np.float32) / 255.
        else: # Assuming color images (classical_sr, real_sr, color_dn, etc.)
            img_lq_bgr = cv2.imread(lq_path, cv2.IMREAD_COLOR)
            img_gt_bgr = cv2.imread(gt_path, cv2.IMREAD_COLOR)
            if img_lq_bgr is None: raise ValueError(f"Could not read LQ image: {lq_path}")
            if img_gt_bgr is None: raise ValueError(f"Could not read GT image: {gt_path}")
            img_lq = cv2.cvtColor(img_lq_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
            img_gt = cv2.cvtColor(img_gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.


        # Transpose to CHW
        img_lq = np.transpose(img_lq, (2, 0, 1))
        img_gt = np.transpose(img_gt, (2, 0, 1))

        return torch.from_numpy(img_lq), torch.from_numpy(img_gt), os.path.basename(lq_path)
# --- END CustomSRDataset Class ---


def main():
    parser = argparse.ArgumentParser()
    # --- EXISTING ARGS ---
    parser.add_argument('--task', type=str, default='color_dn', help='classical_sr, lightweight_sr, real_sr, '
                                                                     'gray_dn, color_dn, jpeg_car, color_jpeg_car')
    parser.add_argument('--scale', type=int, default=1, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')

    # --- ADDED ARGS for Training ---
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'], help='Run in training or testing mode')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--save_dir_train', type=str, default='training_results/swinir_finetuned', help='Directory to save fine-tuned models')
    parser.add_argument('--log_freq', type=int, default=10, help='Frequency to print training log (batches)')
    parser.add_argument('--save_freq_epochs', type=int, default=5, help='Frequency to save model checkpoints (epochs)')
    parser.add_argument('--folder_lq_train', type=str, default=None, help='Input LQ training image folder')
    parser.add_argument('--folder_gt_train', type=str, default=None, help='Input GT training image folder')
    parser.add_argument('--folder_lq_val', type=str, default=None, help='Input LQ validation image folder')
    parser.add_argument('--folder_gt_val', type=str, default=None, help='Input GT validation image folder')

    # Layer Freezing Arguments
    parser.add_argument('--freeze_shallow', action='store_true', help='Freeze conv_first and patch_embed layers')
    parser.add_argument('--freeze_rstb_upto', type=int, default=-1, help='Freeze RSTB layers (model.layers) up to this index (e.g., 0, 1). -1 means no RSTB freezing.')
    # --- END ADDED ARGS ---

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- MODEL DEFINITION (downloads if necessary) ---
    # Ensure model_path points to the PRE-TRAINED model for fine-tuning
    if args.mode == 'train' and not os.path.exists(args.model_path):
         print(f"Warning: Pre-trained model for fine-tuning not found at {args.model_path}. Check path or download.")
         # Decide if you want to attempt download or exit
         # For simplicity, we assume it exists or the download logic in define_model handles it.

    if args.mode == 'test' and not os.path.exists(args.model_path): # Original download logic for testing
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)

    model = define_model(args) # define_model will load weights from args.model_path
    model = model.to(device)


    # --- DISPATCH TO TRAIN OR TEST ---
    if args.mode == 'train':
        print("--- Starting Training Mode ---")
        # --- LAYER FREEZING LOGIC ---
        if args.freeze_shallow:
            for name, param in model.named_parameters():
                # Based on SwinIR's structure in network_swinir.py
                if 'conv_first' in name or \
                   name.startswith('patch_embed.') or \
                   name.startswith('layers.0.patch_embed.') : # PatchEmbed in RSTB uses this name for its own embed
                    param.requires_grad = False
                    print(f"Froze shallow layer: {name}")

        if args.freeze_rstb_upto >= 0:
            if hasattr(model, 'layers') and isinstance(model.layers, nn.ModuleList):
                for i in range(args.freeze_rstb_upto + 1):
                    if i < len(model.layers):
                        for param_name, param in model.layers[i].named_parameters():
                            param.requires_grad = False
                        print(f"Froze parameters in RSTB block: model.layers[{i}]")
                    else:
                        print(f"Warning: RSTB layer index {i} for freezing is out of bounds (max: {len(model.layers)-1}).")
            else:
                print("Warning: Model does not have 'layers' attribute or it's not a ModuleList. Cannot freeze RSTB blocks by index.")

        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters after freezing: {num_trainable_params}")
        if num_trainable_params == 0:
            print("Error: No parameters are trainable. Check freezing logic or model structure.")
            return
        # --- END LAYER FREEZING LOGIC ---

        run_training(args, model, device)

    elif args.mode == 'test':
        print("--- Starting Testing Mode ---")
        run_testing(args, model, device)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


def run_training(args, model, device):
    if not args.folder_lq_train or not args.folder_gt_train:
        print("Error: Training folders (--folder_lq_train and --folder_gt_train) are required for training mode.")
        return

    os.makedirs(args.save_dir_train, exist_ok=True)

    # DataLoaders
    train_dataset = CustomSRDataset(lq_dir=args.folder_lq_train, gt_dir=args.folder_gt_train, scale_factor=args.scale, task=args.task)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    val_dataloader = None
    if args.folder_lq_val and args.folder_gt_val:
        val_dataset = CustomSRDataset(lq_dir=args.folder_lq_val, gt_dir=args.folder_gt_val, scale_factor=args.scale, task=args.task)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True) # Often batch_size=1 for val
        print(f"Using validation set from {args.folder_lq_val} and {args.folder_gt_val}")
    else:
        print("No validation set provided.")


    # Optimizer and Loss
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.L1Loss() # Common for image restoration
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 3, gamma=0.5) # Example scheduler

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for i, (img_lq, img_gt, _) in enumerate(train_dataloader):
            img_lq, img_gt = img_lq.to(device), img_gt.to(device)

            optimizer.zero_grad()
            output = model(img_lq) # SwinIR forward pass handles normalization and padding
            loss = criterion(output, img_gt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % args.log_freq == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_dataloader)}], Batch Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] completed. Average Training Loss: {avg_epoch_loss:.4f}")
        
        # if scheduler:
        #     scheduler.step()

        # Validation step (optional but recommended)
        if val_dataloader:
            model.eval()
            val_psnr_accum = 0.0
            val_ssim_accum = 0.0
            val_loss_accum = 0.0
            with torch.no_grad():
                for img_lq_val, img_gt_val, _ in val_dataloader:
                    img_lq_val, img_gt_val = img_lq_val.to(device), img_gt_val.to(device)
                    output_val = model(img_lq_val)
                    val_loss_accum += criterion(output_val, img_gt_val).item()

                    # Convert to numpy for PSNR/SSIM calculation
                    output_val_np = output_val.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    img_gt_val_np = img_gt_val.data.squeeze().float().cpu().clamp_(0, 1).numpy()

                    if output_val_np.ndim == 3: # CHW -> HWC
                        output_val_np = np.transpose(output_val_np[[2, 1, 0], :, :], (1, 2, 0)) if args.task != 'gray_dn' else np.transpose(output_val_np, (1,2,0))
                        img_gt_val_np = np.transpose(img_gt_val_np[[2, 1, 0], :, :], (1, 2, 0)) if args.task != 'gray_dn' else np.transpose(img_gt_val_np, (1,2,0))
                    
                    output_val_np = (output_val_np * 255.0).round().astype(np.uint8)
                    img_gt_val_np = (img_gt_val_np * 255.0).round().astype(np.uint8)
                    
                    current_psnr = util.calculate_psnr(output_val_np, img_gt_val_np, crop_border=args.scale if args.task in ['classical_sr', 'lightweight_sr', 'real_sr'] else 0)
                    current_ssim = util.calculate_ssim(output_val_np, img_gt_val_np, crop_border=args.scale if args.task in ['classical_sr', 'lightweight_sr', 'real_sr'] else 0)
                    val_psnr_accum += current_psnr
                    val_ssim_accum += current_ssim
            
            avg_val_loss = val_loss_accum / len(val_dataloader)
            avg_val_psnr = val_psnr_accum / len(val_dataloader)
            avg_val_ssim = val_ssim_accum / len(val_dataloader)
            print(f"Epoch [{epoch+1}/{args.epochs}] Validation - Loss: {avg_val_loss:.4f}, PSNR: {avg_val_psnr:.2f} dB, SSIM: {avg_val_ssim:.4f}")


        # Save checkpoint
        if (epoch + 1) % args.save_freq_epochs == 0 or (epoch + 1) == args.epochs:
            checkpoint_name = f'swinir_finetuned_epoch_{epoch+1}.pth'
            if args.task: checkpoint_name = f'{args.task}_' + checkpoint_name
            
            checkpoint_path = os.path.join(args.save_dir_train, checkpoint_name)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'args': args # Save args for reproducibility
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    print("Fine-tuning finished.")


def run_testing(args, model, device):
    model.eval() # Set model to evaluation mode

    # setup folder and path for testing (original logic)
    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnrb'] = []
    test_results['psnrb_y'] = []
    psnr, ssim, psnr_y, ssim_y, psnrb, psnrb_y = 0, 0, 0, 0, 0, 0

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        imgname, img_lq, img_gt = get_image_pair(args, path)
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)

        with torch.no_grad():
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq_padded = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq_padded = torch.cat([img_lq_padded, torch.flip(img_lq_padded, [3])], 3)[:, :, :, :w_old + w_pad]
            
            output = test(img_lq_padded, model, args, window_size) # Original test function call
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        output_np = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output_np.ndim == 3:
            output_np = np.transpose(output_np[[2, 1, 0], :, :], (1, 2, 0))
        output_np = (output_np * 255.0).round().astype(np.uint8)
        cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output_np)

        if img_gt is not None:
            img_gt_eval = (img_gt * 255.0).round().astype(np.uint8)
            img_gt_eval = img_gt_eval[:h_old * args.scale, :w_old * args.scale, ...]
            img_gt_eval = np.squeeze(img_gt_eval)

            psnr = util.calculate_psnr(output_np, img_gt_eval, crop_border=border)
            ssim = util.calculate_ssim(output_np, img_gt_eval, crop_border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if img_gt_eval.ndim == 3:
                psnr_y = util.calculate_psnr(output_np, img_gt_eval, crop_border=border, test_y_channel=True)
                ssim_y = util.calculate_ssim(output_np, img_gt_eval, crop_border=border, test_y_channel=True)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
            if args.task in ['jpeg_car', 'color_jpeg_car']:
                psnrb = util.calculate_psnrb(output_np, img_gt_eval, crop_border=border, test_y_channel=False)
                test_results['psnrb'].append(psnrb)
                if args.task in ['color_jpeg_car']: # Note: original script had 'color_jpeg_car' for psnrb_y
                    psnrb_y = util.calculate_psnrb(output_np, img_gt_eval, crop_border=border, test_y_channel=True)
                    test_results['psnrb_y'].append(psnrb_y)
            print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNRB: {:.2f} dB;'
                  'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; PSNRB_Y: {:.2f} dB.'.
                  format(idx, imgname, psnr, ssim, psnrb, psnr_y, ssim_y, psnrb_y))
        else:
            print('Testing {:d} {:20s}'.format(idx, imgname))

    if img_gt is not None and test_results['psnr']: # Check if any results were added
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print(f'\n{save_dir} \n-- Average PSNR/SSIM(RGB): {ave_psnr:.2f} dB; {ave_ssim:.4f}')
        if test_results['psnr_y']:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print(f'-- Average PSNR_Y/SSIM_Y: {ave_psnr_y:.2f} dB; {ave_ssim_y:.4f}')
        if test_results['psnrb'] and args.task in ['jpeg_car', 'color_jpeg_car']:
            ave_psnrb = sum(test_results['psnrb']) / len(test_results['psnrb'])
            print(f'-- Average PSNRB: {ave_psnrb:.2f} dB')
            if test_results['psnrb_y'] and args.task in ['color_jpeg_car']:
                ave_psnrb_y = sum(test_results['psnrb_y']) / len(test_results['psnrb_y'])
                print(f'-- Average PSNRB_Y: {ave_psnrb_y:.2f} dB')
    print("Testing finished.")


def define_model(args):
    # This function defines the model structure based on args.task
    # AND loads pre-trained weights from args.model_path if it exists.
    # This behavior is suitable for both testing and fine-tuning.

    # 001 classical image sr
    if args.task == 'classical_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'
    # 002 lightweight image sr
    elif args.task == 'lightweight_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'
    # 003 real-world image sr
    elif args.task == 'real_sr':
        upsampler = 'nearest+conv'
        in_chans = 3
        # training_patch_size is not directly used by model constructor for img_size
        # but kept for consistency with how args were used.
        # img_size in SwinIR constructor is more like a placeholder if patch_size=1
        if not args.large_model:
            model = net(upscale=args.scale, in_chans=in_chans, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler=upsampler, resi_connection='1conv')
        else:
            model = net(upscale=args.scale, in_chans=in_chans, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler=upsampler, resi_connection='3conv')
        param_key_g = 'params_ema'
    # 004 grayscale image denoising
    elif args.task == 'gray_dn':
        model = net(upscale=1, in_chans=1, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'
    # 005 color image denoising
    elif args.task == 'color_dn':
        model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'
    # 006 JPEG compression artifact reduction (grayscale)
    elif args.task == 'jpeg_car':
        model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'
    # 006 JPEG compression artifact reduction (color)
    elif args.task == 'color_jpeg_car':
        model = net(upscale=1, in_chans=3, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'
    else:
        raise ValueError(f"Task {args.task} not recognized in define_model.")

    # Load pre-trained weights if model_path is provided and exists
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model weights from {args.model_path}")
        try:
            # Load checkpoint
            pretrained_model_chkp = torch.load(args.model_path, map_location='cpu')

            # Determine the actual state_dict
            if param_key_g in pretrained_model_chkp:
                actual_state_dict = pretrained_model_chkp[param_key_g]
            elif 'model_state_dict' in pretrained_model_chkp: # Common for custom saved checkpoints
                 actual_state_dict = pretrained_model_chkp['model_state_dict']
            elif 'params' in pretrained_model_chkp: # Fallback if param_key_g was wrong but 'params' exists
                actual_state_dict = pretrained_model_chkp['params']
            else: # Assume the checkpoint is the state_dict itself
                actual_state_dict = pretrained_model_chkp
            
            # It's good practice to filter for keys present in the current model structure
            # and match shapes, especially if fine-tuning might involve slight architectural changes
            # or if loading from different SwinIR versions.
            model_dict = model.state_dict()
            # 1. Filter out unnecessary keys from loaded_state_dict
            filtered_state_dict = {k: v for k, v in actual_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            # 2. Overwrite entries in the existing state dict
            model_dict.update(filtered_state_dict)
            # 3. Load the new state dict
            model.load_state_dict(model_dict, strict=False) # Use strict=False to allow partial loads if shapes/keys mismatch
            
            loaded_keys = len(filtered_state_dict)
            total_keys_model = len(model.state_dict())
            total_keys_chkpt = len(actual_state_dict)
            print(f"Loaded {loaded_keys} / {total_keys_model} matching keys from checkpoint (which had {total_keys_chkpt} keys).")
            if loaded_keys == 0 and total_keys_chkpt > 0 :
                 print("Warning: No weights were loaded. Check model_path, param_key_g, or checkpoint structure.")

        except Exception as e:
            print(f"Error loading model weights: {e}. Model will be initialized randomly or with default SwinIR init.")
    elif args.model_path: # Path provided but does not exist
        print(f"Warning: Model path {args.model_path} not found. Model will be initialized randomly or with default SwinIR init.")
    else: # No model path provided
        print("No pre-trained model path provided. Model will be initialized randomly or with default SwinIR init.")

    return model


def setup(args):
    # (Original setup function - mostly for testing paths)
    if args.task in ['classical_sr', 'lightweight_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        folder = args.folder_gt
        border = args.scale
        window_size = 8
    elif args.task in ['real_sr']:
        save_dir = f'results/swinir_{args.task}_x{args.scale}'
        if args.large_model:
            save_dir += '_large'
        folder = args.folder_lq
        border = 0
        window_size = 8
    elif args.task in ['gray_dn', 'color_dn']:
        save_dir = f'results/swinir_{args.task}_noise{args.noise}'
        folder = args.folder_gt
        border = 0
        window_size = 8
    elif args.task in ['jpeg_car', 'color_jpeg_car']:
        save_dir = f'results/swinir_{args.task}_jpeg{args.jpeg}'
        folder = args.folder_gt
        border = 0
        window_size = 7
    else: # Fallback for unknown tasks, or if called during training setup where only window_size might be needed
        print(f"Warning: Task {args.task} not fully recognized in setup. Using default window_size=8.")
        save_dir = f'results/swinir_unknown_task'
        folder = args.folder_gt if args.folder_gt else args.folder_lq # Or some default
        border = 0
        window_size = 8 # SwinIR default window_size

    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    # (Original get_image_pair function - for testing data loading)
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    if args.task in ['classical_sr', 'lightweight_sr']:
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_lq_path = os.path.join(args.folder_lq, f'{imgname}x{args.scale}{imgext}')
        if not os.path.exists(img_lq_path): # try other common naming for LQ
            img_lq_path = os.path.join(args.folder_lq, f'{imgname}{imgext}') # if LQ has same name
        img_lq = cv2.imread(img_lq_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        if img_lq is None: raise FileNotFoundError(f"LQ image not found at {img_lq_path} or {os.path.join(args.folder_lq, f'{imgname}{imgext}')}")
    elif args.task in ['real_sr']:
        img_gt = None
        img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        if img_lq is None: raise FileNotFoundError(f"LQ image not found at {path}")
    elif args.task in ['gray_dn']:
        img_gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        if img_gt is None: raise FileNotFoundError(f"GT image not found at {path}")
        np.random.seed(seed=0) # for reproducibility
        img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)
        img_gt = np.expand_dims(img_gt, axis=2)
        img_lq = np.expand_dims(img_lq, axis=2)
    elif args.task in ['color_dn']:
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        if img_gt is None: raise FileNotFoundError(f"GT image not found at {path}")
        np.random.seed(seed=0) # for reproducibility
        img_lq = img_gt + np.random.normal(0, args.noise / 255., img_gt.shape)
    elif args.task in ['jpeg_car']:
        img_gt_orig = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img_gt_orig is None: raise FileNotFoundError(f"GT image not found at {path}")
        if img_gt_orig.ndim != 2:
            img_gt_y = util.bgr2ycbcr(img_gt_orig, y_only=True)
        else:
            img_gt_y = img_gt_orig
        result, encimg = cv2.imencode('.jpg', img_gt_y, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
        img_lq_y = cv2.imdecode(encimg, 0) # 0 for grayscale
        img_gt = np.expand_dims(img_gt_y, axis=2).astype(np.float32) / 255.
        img_lq = np.expand_dims(img_lq_y, axis=2).astype(np.float32) / 255.
    elif args.task in ['color_jpeg_car']:
        img_gt_orig = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_gt_orig is None: raise FileNotFoundError(f"GT image not found at {path}")
        result, encimg = cv2.imencode('.jpg', img_gt_orig, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
        img_lq_bgr = cv2.imdecode(encimg, 1) # 1 for color
        img_gt = img_gt_orig.astype(np.float32)/ 255.
        img_lq = img_lq_bgr.astype(np.float32)/ 255.
    else:
        raise ValueError(f"Task {args.task} data loading not implemented in get_image_pair.")

    return imgname, img_lq, img_gt


def test(img_lq, model, args, window_size):
    # (Original test function for inference during testing)
    if args.tile is None:
        output = model(img_lq)
    else:
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)
                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)
    return output

if __name__ == '__main__':
    main()