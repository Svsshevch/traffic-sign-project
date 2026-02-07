# Traffic Sign Detection and Classification

A two-stage deep learning pipeline for traffic sign detection and classification using YOLOv8 and custom CNN.

## Project Overview

This project implements a complete traffic sign recognition system consisting of:

1. **Detection Model** (`Traffic_Sign_Detection.ipynb`): YOLOv8-based detector trained on GTSDB dataset to localize traffic signs
2. **Classification Model** (`Traffic_Sign_Classificator.ipynb`): Custom CNN trained on GTSRB dataset to classify 43 types of traffic signs

The system processes video frames by first detecting traffic sign regions with YOLO, then classifying each detected region using the CNN.

---

### Required Libraries
```bash
pip install torch torchvision
pip install ultralytics  # for YOLOv8
pip install opencv-python
pip install numpy pandas matplotlib seaborn
pip install pillow scikit-learn tqdm
pip install wandb  # for experiment tracking
```
### Weights & Biases (W&B) Setup
W&B is used for experiment tracking. Setup is required before training:

1. **Create W&B Account (if you don't have one)**
2. **Get API Key**
3. **Login to W&B**

---

## 1. Classification Model Training 

This notebook trains the CNN classifier on the GTSRB dataset.

**Steps:**

1. **Download Dataset**
   ```bash
   wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip
   unzip -qq GTSRB_Final_Training_Images.zip
   ```

2. **Data Preparation**
   - The notebook automatically organizes images into train/val/test splits (80/10/10)
   - Applies data augmentation to balance underrepresented classes

3. **Training**
   - Run cells sequentially to train the CNN model
  
4. **Output**
   - Model saved as `my_traffic_sign_model.pth`
   - Best checkpoint saved during training
   - Achieves ~99.74% test accuracy

---

## 2. Detection Model Training and Inference

This notebook trains the YOLO detector and runs the complete detection + classification pipeline.

**Steps:**

1. **Download GTSDB Dataset**
   ```bash
   wget https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip
   unzip -q FullIJCNN2013.zip
   mv FullIJCNN2013 GTSDB
   ```

2. **Prepare Dataset for YOLO**
   - Run the dataset cells
   - Creates YOLO-format annotations (bounding boxes)
   - Splits data into train/val sets

3. **Train YOLO Detector**
   - The notebook trains YOLOv8 on single class (traffic signs)

4. **Load Pre-trained CNN Classifier**
   - Upload your trained CNN model (`my_traffic_sign_model.pth`)
   - Model loaded with 43 output classes

5. **Run Inference**

   **Option A: Webcam Detection**
   ```python
   webcam_detection()
   ```
   - Real-time detection from webcam
   - Press 'q' to quit

   **Option B: Video Processing**
   ```python
   process_videos(video_path="your_video.mp4")
   ```
   - Processes video file
   - Outputs annotated video with detections


## How the Pipeline Works

1. **YOLO Detection**: Detects traffic sign bounding boxes in each frame
2. **Region Extraction**: Crops detected regions from the frame
3. **CNN Classification**: Classifies each cropped region into 43 classes
4. **Confidence Filtering**: Only displays detections where both YOLO and CNN confidence > 0.7
5. **Visualization**: Draws bounding boxes and labels on frame

