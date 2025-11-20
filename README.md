# Pest_Detection
This project aims to implement crop pest object detection based on Faster R-CNN (with ResNet-50-FPN as the backbone network) using PyTorch. The project includes a baseline model and an improved version, the latter of which adopts a more refined training strategy, focusing on enhancing the accuracy of category recognition.
## 1. Project structure
```
Pest_Detection/
└── CNN/
    ├── Faster_R-CNN/              
    │   ├── runs/                
    │   ├── Faster_R-CNN.py    
    │   ├── test.py               
    │   └── train.py             
    ├── Faster_R-CNN_Improved/   
    │   ├── runs/                
    │   ├── best_faster_rcnn_weights.pt
    │   ├── Faster_R-CNN_Improved.py
    │   ├── test.py               
    │   └── train.py               
    ├── output/                  
    ├── data.yaml                  
    └── detect.py
```
## 2. Environmental Requirements and Dependencies
This project is based on the PyTorch deep learning framework and uses torchmetrics and sklearn for evaluation, as well as albumentations for data augmentation.
```
pip install torch torchvision torchaudio 
pip install numpy opencv-python Pillow
pip install torchmetrics scikit-learn albumentations matplotlib tensorboard
```
| Category | Library | Purpose | Installation Command (suggested) |
| :--- | :--- | :--- | :--- |
| **Deep Learning** | **`torch`** | Core PyTorch framework. | `pip install torch torchvision torchaudio` |
| | **`torchvision`** | Contains the Faster R-CNN model, pre-trained weights, and image transformations. | (Included with `torch`) |
| **Data Handling** | **`numpy`** | Fundamental package for numerical operations and array handling. | `pip install numpy` |
| | **`opencv-python`** | Used by the `AgroPestYOLODataset` class for image loading and BGR to RGB conversion. | `pip install opencv-python` |
| **Augmentation** | **`albumentations`** | Used for efficient and flexible data augmentation pipelines in the `AgroPestYOLODataset`. | `pip install albumentations` |
| **Evaluation** | **`torchmetrics`** | Used to calculate **mAP** metrics during validation and testing. | `pip install torchmetrics` |
| | **`scikit-learn`** | Used to calculate standard classification metrics (Precision, Recall, F1, Accuracy) in the `evaluate_model` function. | `pip install scikit-learn` |
| **Logging/Vis** | **`tensorboard`** | Used by `train.py` for tracking training loss and validation metrics. | `pip install tensorboard` |
| | **`matplotlib`** | Likely required by `detect.py` for displaying inference results. | `pip install matplotlib` |
| **Other** | **`Pillow`** (PIL) | Common dependency for image processing libraries. | `pip install Pillow` |

## 3. Model Training
### 3.1 Baseline Model (Faster_R-CNN.py)
* **Configuration**: Strategy
    * **Backbone**: ResNet-50-FPN
    * **Optimizer**: SGD (LR: 0.005, Momentum: 0.9, Weight Decay: $5 \times 10^{-4}$)
    * **LR Scheduler**: StepLR (LR drops by a factor of 10 every 3 epochs)
    * **Data Augmentation**: Includes geometric transformations (flip, rotation, scale) and pixel-level transformations
### 3.2 Improved Model (Faster_R-CNN_Improved.py)
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
python Faster_R-CNN_Improved\train.py
```
### Running Testing
```
# Run testing for the Improved Model (Logs indicate you ran this version)
python Faster_R-CNN_Improved\test.py
```
## 4. Improved Model Training Results Summary (30 Epochs)
| Metric | Best Validation Performance (Epoch 13) | Final Test Set Performance |
| :--- | :---: | :---: |
| **Detection mAP@0.5** | 0.7352 | **0.7476** |
| **mAP (COCO)** | N/A | **0.3987** |
| **F1 Score (TP-based)** | 0.7920 | **0.7713** |
| **Total Training Time** | N/A | 13h 30m 35s |
## 5. Model Inference and Visualization (detect.py)
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
