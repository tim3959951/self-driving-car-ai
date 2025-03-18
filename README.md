# 🚗 Self-Driving Cars Controlled by Deep Learning  

🏆 **Kaggle Competition (2nd Place) | Deep Learning | Computer Vision**  

This project builds a self-driving car control system using **an ensemble of EfficientNetB5 & EfficientNetB7 CNNs**, trained on a dataset of steering angles and speeds. For real-time self-driving applications, a **MobileNet model optimized with TensorFlow Lite & Edge TPU** enhances inference efficiency. To improve robustness, the system employs **K-Fold Cross-Validation and Image Augmentation**. The model is designed to navigate complex driving scenarios, including **traffic lights, roundabouts, figure-eight tracks, T-junctions, pedestrians, and obstacles**, ensuring safe and efficient autonomous driving.

---

## 📌 **Project Overview**
- **Task**: Predict steering angle and speed from images captured by a front-facing car camera.
- **Dataset**: 13K training images with corresponding steering angles and speeds.
- **Models**: CNN-based architecture (**EfficientNet B5 & B7**).
- **Optimisation**: Learning rate scheduling, dropout, and hyperparameter tuning.
- **Deployment**: TensorFlow Lite, Edge TPU for real-time inference.

---


## 📂 Project Structure

| File/Folder                      | Description |
|----------------------------------|--------------------------------------------------|
| 📂 `src`                         | Model training & inference scripts |
| 📂 `experiments`                 | Model experiments & evaluation |
| 📄 `ensemble_kfold_imgaug.ipynb` | Full training pipeline notebook |
| 📄 `evaluate_h5_model.ipynb`         | Data exploration & visualisation |
| 📄 `requirements.txt`            | Dependencies |
| 📄 `README.md`                   | Project documentation |
| 📄 `.gitignore`                  | Ignore unnecessary files |






# Project documentation
---

## 🚀 **Main Technologies**
- **Deep Learning**: TensorFlow, Keras, EfficientNet (B5, B7)
- **Computer Vision**: OpenCV, Image Augmentation (ImgAug)
- **Training Techniques**: K-Fold Cross-Validation, Hyperparameter Tuning
- **Optimisation**: Learning Rate Scheduling, Model Pruning, Quantisation
- **MLOps & Deployment**: TensorFlow Lite, Edge TPU

---
## 🔬 Model Experiments  

We tested multiple deep learning architectures and training strategies to optimise performance. Below is a summary of key results:  

| Model                     | Validation Accuracy | MSE  | Key Techniques |
|---------------------------|--------------------|------|---------------|
| **EfficientNetB5 + B7 (Ensemble)** | **97.32%** | **0.0131** | Ensembling, K-Fold, Augmentation, Learning Rate Scheduling |
| **MobileNetV2 (K-Fold Validation)** | **97.25%** | **0.0134** | K-Fold, Augmentation, Advanced Data Preprocessing |
| **EfficientNetB7 (Baseline)** | **97.00%** | **0.0138** | Baseline Training, Custom Weight Initialisation |
| **EfficientNetB2 (K-Fold + Aug)** | **96.50%** | **0.0157** | K-Fold, Augmentation, Label Smoothing |

📌 **Key Insights**:  
✅ **EfficientNetB5 + B7 (Ensemble)** achieved the highest accuracy & lowest error.  
✅ **K-Fold Cross-Validation & Augmentation** significantly improved performance.  
✅ **MobileNetV2** offers **faster inference**, making it ideal for deployment.  

For **detailed experiment results**, including more models & training logs, check the full notebook:  
📄 [Full Experiment Results](./experiments)



---

## 🔧 **Training & Inference**
### 🏋️‍♂️ **Train the Model**
```bash
python src/train.py

🏎 Run Inference on Test Images

python src/inference.py --model best_model.h5 --test_data data/test_data/

📊 Results
Metric	Value
MSE	0.01398
MAE (Angle)	0.021
MAE (Speed)	0.018
✅ Final model achieves state-of-the-art accuracy with 2nd place in Kaggle competition!

📦 Installation & Dependencies

pip install -r requirements.txt


