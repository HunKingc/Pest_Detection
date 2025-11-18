# Pest_Detection
This project aims to implement crop pest object detection based on Faster R-CNN (with ResNet-50-FPN as the backbone network) using PyTorch. The project includes a baseline model and an improved version, the latter of which adopts a more refined training strategy, focusing on enhancing the accuracy of category recognition.
## 1. Project structure
```
Pest_Detection/
├── Faster_R-CNN.py           # 基线模型：包含 StepLR 和通用数据增强
├── Faster_R-CNN_Improved.py  # 改进模型：包含 CosineAnnealingLR、梯度裁剪和像素级增强
├── detect.py                 # 模型推理与可视化脚本
├── data.yaml                 # 数据集配置（类别名称）
├── best_faster_rcnn_weights.pth # (训练后生成) 最佳模型权重，保存在项目根目录
├── train/
├── valid/
└── test/
```
## 2. Environmental Requirements and Dependencies
This project is based on the PyTorch deep learning framework and uses torchmetrics and sklearn for evaluation, as well as albumentations for data augmentation.
```
# 建议使用 Conda/Virtualenv 管理环境
pip install torch torchvision torchaudio  # 根据您的 CUDA 版本调整
pip install numpy opencv-python Pillow
pip install torchmetrics scikit-learn albumentations matplotlib tensorboard
```
## 3. Dataset category configuration (Faster R-CNN labels)
The model detects a total of 12 categories. In the Faster R-CNN labeling convention, label 0 represents the background (background), and the actual categories start from label 1. 
Please note: The category names in the training files (such as "Aphids", "Armyworm") are inconsistent with those in data.yaml (such as "Ants", "Bees"). When running the inference script detect.py, the following corrected list must be used.

running the inference script `detect.py`, **you must** use the following corrected list.

| Faster R-CNN Label | Class Name (from data.yaml) | Corresponding Training Output Class (For reference only) |
| :---: | :---: | :---: |
| 0 | `background` | N/A |
| 1 | `Ants` | Aphids (Training Output) |
| 2 | `Bees` | Armyworm (Training Output) |
| 3 | `Beetles` | Beetle (Training Output) |
| 4 | `Caterpillars` | Bollworm (Training Output) |
| 5 | `Earthworms` | Grasshopper (Training Output) |
| 6 | `Earwigs` | Leaf_Miner (Training Output) |
| 7 | `Grasshoppers` | Mealybug (Training Output) |
| 8 | `Moths` | Mite (Training Output) |
| 9 | `Slugs` | Moth (Training Output) |
| 10 | `Snails` | Scale (Training Output) |
| 11 | `Wasps` | Thrips (Training Output) |
| 12 | `Weevils` | Whitefly (Training Output) |
### Update CLASS_NAMES in detect.py to the following list:
```
CLASS_NAMES = [
    'background', 'Ants', 'Bees', 'Beetles', 'Caterpillars', 'Earthworms',
    'Earwigs', 'Grasshoppers', 'Moths', 'Slugs', 'Snails', 'Wasps', 'Weevils'
]
```
## 4. Model Training
### 4.1 Baseline Model (Faster_R-CNN.py)
* **Configuration**: Strategy
    * **Backbone**: ResNet-50-FPN
    * **Optimizer**: SGD (LR: 0.005, Momentum: 0.9, Weight Decay: $5 \times 10^{-4}$)
    * **LR Scheduler**: StepLR (LR drops by a factor of 10 every 3 epochs)
    * **Data Augmentation**: Includes geometric transformations (flip, rotation, scale) and pixel-level transformations
### 4.2 Improved Model (Faster_R-CNN_Improved.py)
* **Configuration**: Strategy
    * **Backbone**: ResNet-50-FPN
    * **Optimizer**: SGD (LR: 0.001, Momentum: 0.9, Weight Decay: $5 \times 10^{-4}$)
    * **LR Scheduler**: CosineAnnealingLR (Min LR: $1 \times 10^{-6}$)
    * **Data Augmentation**: Only pixel-level transformations retained (removed geometric to stabilize bounding box regression)
    * **Gradient Clipping**: `max_norm = 5.0`
### Running Training
The training script saves the model weights with the highest mAP@0.5 on the validation set to best_faster_rcnn_weights.pth.
```
# Run training for the Improved Model (Logs indicate you ran this version)
python Faster_R-CNN_Improved.py
```
## 5. Improved Model Training Results Summary (30 Epochs)
| Metric | Best Validation Performance (Epoch 13) | Final Test Set Performance |
| :--- | :---: | :---: |
| **Detection mAP@0.5** | 0.7352 | **0.7476** |
| **mAP (COCO)** | N/A | **0.3987** |
| **F1 Score (TP-based)** | 0.7920 | **0.7713** |
| **Total Training Time** | N/A | 13h 30m 35s |
## 6. Model Inference and Visualization (detect.py)
detect.py is used to load the trained weights and perform detection and visualization on a single image.

Running Inference
Verify Weights: Ensure best_faster_rcnn_weights.pth is present in the project root directory.

Update Class Names: Make sure CLASS_NAMES in detect.py is updated as required in Section 3.

Specify Image Path: Modify test_image_path at the bottom of the detect.py script.
```
# Example: Run inference and display results
python detect.py
```
The script will load the model, perform inference using the predefined threshold (default is 0.5), and display the result in a Matplotlib window.
