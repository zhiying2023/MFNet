# MFNet

This project utilizes the **MFNet framework** to conduct salient object detection experiments with multiple backbone networks.

> **Note**: The network code will be released upon acceptance.

## 📋 Environment Setup

Ensure the following dependencies are installed:

- **CUDA**: 11.7
- **Python**: 3.10.11
- **PyTorch**: 2.0.0

We recommend using **Conda** for environment management.

## 🛠 Installation Instructions

### 1. Install MambaVision Dependencies

Download and install the required packages:

- **Causal-Conv1d**  
  Download from: [causal-conv1d GitHub](https://github.com/Dao-AILab/causal-conv1d/releases)  
  Install the wheel file:
  ```bash
  pip install causal_conv1d-1.4.0+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
  ```

- **Mamba-SSM**  
  Download from: [Mamba-SSM GitHub](https://github.com/state-spaces/mamba/releases/tag/v1.2.2)  
  Install the wheel file:
  ```bash
  pip install mamba_ssm-1.2.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
  ```

### 2. Install MMCV

Install **MMCV** compatible with your CUDA, Python, and PyTorch versions. Refer to the official documentation:  
👉 [MMCV Installation Guide](https://mmcv.readthedocs.io/zh-cn/latest/get_started/installation.html)

> ⚠️ **Warning**: Ensure compatibility between Python, CUDA, PyTorch, and MMCV versions to avoid errors.

## 📂 Dataset Download

Download the datasets from one of the following sources:  
- **GitHub**: [LFSOD-Survey](https://github.com/kerenfu/LFSOD-Survey)  
- **Baidu Netdisk**: [Link](https://pan.baidu.com/s/1WAFxO8qmeQuTSuDnlOmGGg?pwd=lfso) (Password: `lfso`)

The datasets include **DUTLF-FS**, **HFUT**, and **Lytro-Illum** at 256×256 resolution.  
- **Training Data**: Focal stack images, depth images, all-focus images, inner/outer boundary images, and ground truth masks.  
- **Testing Data**: Focal stack images, depth images, all-focus images, and ground truth masks.

### Dataset Structure

```
dataset/
├── train/
│   ├── DUTLF-FS/
│   │   ├── train_focals/
│   │   ├── train_images/
│   │   ├── train_depths/
│   │   ├── train_edge_external/
│   │   ├── train_edge_internal/
│   │   ├── train_masks/
│   ├── HFUT/
│   ├── Lytro-Illum/
├── test/
│   ├── DUTLF-FS/
│   │   ├── test_focals/
│   │   ├── test_images/
│   │   ├── test_depths/
│   │   ├── test_masks/
│   ├── HFUT/
│   ├── Lytro-Illum/
```

## 🔗 Pretrained Backbone Weights

Download pretrained backbone weights from:  
- **GitHub**: [NVlabs/MambaVision](https://github.com/NVlabs/MambaVision)  
- **Baidu Netdisk**: [Link](https://pan.baidu.com/s/1WB1yp5x4vdEvBMDBGWkkPg?pwd=lfso) (Password: `lfso`)

Place the downloaded weights in:  
```
./pretrained_params/
```

## 🖼️ Saliency Map Results

Download saliency map results from:  
- **Baidu Netdisk**: [Link](https://pan.baidu.com/s/12YRAm2zYIELGmf6KskhUxA?pwd=lfso) (Password: `lfso`)

---

Let me know if you'd like further refinements or additional sections!