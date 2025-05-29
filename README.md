# Fine-tuned SwinIR for Custom Image Super-Resolution

This project is a fork of the original **SwinIR: Image Restoration Using Swin Transformer** implementation. It has been adapted and fine-tuned for a custom image super-resolution dataset. Key modifications include the addition of training capabilities, layer freezing options, and custom dataset handling.

**Original Project:** [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)

## Key Features & Modifications

*   **Fine-tuning:** The `main_test_swinir.py` script has been significantly augmented to support training (fine-tuning) the SwinIR model on custom datasets.
*   **Custom Dataset:** Includes a `CustomSRDataset` class to load Low-Resolution (LR) and High-Resolution (HR) image pairs from specified directories.
*   **Layer Freezing:** Implemented options to freeze specific parts of the SwinIR model during fine-tuning:
    *   `--freeze_shallow`: Freezes the initial `conv_first` and `patch_embed` layers.
    *   `--freeze_rstb_upto N`: Freezes Residual Swin Transformer Blocks (RSTB) up to the N-th index (e.g., `--freeze_rstb_upto 1` freezes the first two RSTBs: index 0 and 1).
    *   `--freeze_reconstruction`: Freezes the final reconstruction/upsampling layers (e.g., `conv_last`, upsampler modules).
*   **Test-time Visualization:** Option to display/save LR-SR-HR image triplets during testing for qualitative assessment.
*   **Dataset Preprocessing (User Workflow):**
    *   **Image Rotation for Consistency:** My custom dataset images were preprocessed using a separate Python script to ensure all images are in a consistent landscape orientation. This step is performed *before* using them with this codebase.
    *   **Augmentation:** During my dataset preparation, basic flipping (horizontal/vertical) was considered as an offline augmentation strategy. The provided `CustomSRDataset` currently does not implement on-the-fly augmentations beyond basic loading and normalization, but this could be extended.

## File Structure

*   `main_test_swinir.py`: The main script for both training and testing. Contains argument parsing, model definition, data loading, training loop, testing loop, and layer freezing logic.
*   `models/network_swinir.py`: The core SwinIR model architecture (largely based on the original).
*   `utils/util_calculate_psnr_ssim.py`: Utility functions for calculating PSNR and SSIM metrics (from original).
*   `training_results/`: (Example) Directory where fine-tuned models will be saved by default during training.
*   `results/`: (Example) Directory where test outputs (super-resolved images, metrics) will be saved.

## Prerequisites

*   Python 3.x
*   PyTorch (tested with 1.9+, CUDA recommended for speed)
*   OpenCV (`opencv-python`)
*   NumPy
*   Matplotlib
*   Requests (for downloading pre-trained models)
*   TIMM (`timm`)

You can install most of these via pip:
```bash
pip install torch torchvision torchaudio opencv-python numpy matplotlib requests timm
```
## Data Preparation
<pre><code>
your_dataset_root/
├── train/
│   ├── LR/
│   │   ├── image001.png
│   │   └── image002.png
│   └── HR/
│       ├── image001.png
│       └── image002.png
├── valid/
│   ├── LR/
│   │   ├── image101.png
│   │   └── image102.png
│   └── HR/
│       ├── image101.png
│       └── image102.png
└── test/
    ├── LR/
    │   ├── image201.png
    │   └── image202.png
    └── HR/
        ├── image201.png
        └── image202.png
</code></pre>

Note: For tasks like classical_sr, the folder_lq often contains images like image001x4.png if image001.png is the HR name and scale is 4. The get_image_pair function and CustomSRDataset handle file naming conventions based on the task.
(Optional, Recommended for your workflow) As mentioned, preprocess your images (e.g., using your custom Python script) to ensure they are in a consistent orientation (e.g., landscape) before placing them into the LR/HR folders.
## Usage
The script main_test_swinir.py can be run from the command line. It supports both training (--mode train) and testing (--mode test).
## Training
Here's an example command for fine-tuning, particularly suited for notebook environments like Kaggle or Colab. This command fine-tunes a pre-trained SwinIR model for classical super-resolution (x4), freezing the first two RSTB blocks.
```bash
!python main_test_swinir.py \
    --mode train \
    --task classical_sr \
    --scale 4 \
    --model_path /kaggle/input/the-model/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth \
    --folder_lq_train /kaggle/input/swinirdata/pro_dataset/train/LR \
    --folder_gt_train /kaggle/input/swinirdata/pro_dataset/train/HR \
    --folder_lq_val /kaggle/input/swinirdata/pro_dataset/valid/LR \
    --folder_gt_val /kaggle/input/swinirdata/pro_dataset/valid/HR \
    --epochs 20 \
    --batch_size 4 \
    --lr 1e-5 \
    --save_dir_train /kaggle/working/fine_tuned_models \
    --freeze_rstb_upto 1
```
Key Training Arguments:
--mode train: Sets the script to training mode.
--task classical_sr: Specifies the super-resolution task.
--scale 4: Sets the desired upscaling factor.
--model_path: Path to the pre-trained model weights to start fine-tuning from.
--folder_lq_train, --folder_gt_train: Paths to your training LR and HR images.
--folder_lq_val, --folder_gt_val: Paths to your validation LR and HR images (optional but recommended).
--epochs: Number of training epochs.
--batch_size: Training batch size.
--lr: Learning rate.
--save_dir_train: Directory to save the fine-tuned model checkpoints.
--freeze_rstb_upto 1: Example of layer freezing. Freezes RSTB layers 0 and 1.
Other freezing options: --freeze_shallow, --freeze_reconstruction.
Modify paths and hyperparameters as needed for your setup and dataset.
## Testing
After training, or to test a pre-trained/fine-tuned model, use the test mode. This example tests a fine-tuned model and displays the first 20 LR-SR-HR image triplets.
```bash
!python main_test_swinir.py \
    --mode test \
    --task classical_sr \
    --scale 4 \
    --model_path /kaggle/working/fine_tuned_models/classical_sr_swinir_finetuned_epoch_20.pth \
    --folder_lq /kaggle/input/swinirdata/pro_dataset/test/LR \
    --folder_gt /kaggle/input/swinirdata/pro_dataset/test/HR \
    --display_n_triplets 20
```
Key Testing Arguments:
--mode test: Sets the script to testing mode.
--task classical_sr: Specifies the task (should match the model).
--scale 4: Upscaling factor (should match the model).
--model_path: Path to the trained model weights to be tested.
--folder_lq: Path to the low-quality test images.
--folder_gt: Path to the ground-truth (high-quality) test images (optional, for metrics calculation and display).
--display_n_triplets 20: Number of LR-SR-GT image triplets to display (plots will be saved in the results directory). Set to 0 to disable.
--tile / --tile_overlap: For testing large images by processing them in tiles (optional).
The default output directory for test results is results/swinir_<task_details>. Visualizations will also be saved there.
## Acknowledgements
This work is based on the official SwinIR implementation by Jingyun Liang et al.
Swin Transformer: Liu, Ze, et al. "Swin transformer: Hierarchical vision transformer using shifted windows." arXiv preprint arXiv:2103.14030 (2021).
SwinIR: Liang, Jingyun, et al. "SwinIR: Image restoration using Swin Transformer." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
## License
This project inherits the license of the original SwinIR project if applicable. Please refer to the original repository for specific license details. Typically, research code like this is released under a permissive license like MIT or Apache 2.0.
