# 🚗 Self-Driving Cars Controlled by Deep Learning  

🏆 **Kaggle Competition (2nd Place) | Deep Learning | Computer Vision**  

This project builds a self-driving car control system using **EfficientNetB5 & B7 CNNs**, trained on a dataset of steering angles and speeds. The model optimises real-time inference with **TensorFlow Lite & Edge TPU**, while employing **K-Fold Cross-Validation and Image Augmentation** for robustness.

---

## 📌 **Project Overview**
- **Task**: Predict steering angle and speed from images captured by a front-facing car camera.
- **Dataset**: 13K training images with corresponding steering angles and speeds.
- **Models**: CNN-based architecture (**EfficientNet B5 & B7**).
- **Optimisation**: Learning rate scheduling, dropout, and hyperparameter tuning.
- **Deployment**: TensorFlow Lite, Edge TPU for real-time inference.

---

## 📂 **Project Structure**

📂 self-driving-car-ai

│── 📂 src      
│   ├── model.py           
│   ├── data_loader.py    
│   ├── augmentation.py    
│   ├── train.py           
│   ├── inference.py      
│   ├── data_analysis.py   
│── **ensemble_kfold_imgaug.ipynb**   # Full training process in a notebook
│── **data_analysis.ipynb**   
│── requirements.txt       
│── README.md              
│── .gitignore     

# Project documentation
---

## 🚀 **Main Technologies**
- **Deep Learning**: TensorFlow, Keras, EfficientNet (B5, B7)
- **Computer Vision**: OpenCV, Image Augmentation (ImgAug)
- **Training Techniques**: K-Fold Cross-Validation, Hyperparameter Tuning
- **Optimisation**: Learning Rate Scheduling, Model Pruning, Quantisation
- **MLOps & Deployment**: TensorFlow Lite, Edge TPU, Docker

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
📄 [Full Experiment Results](./experiments.ipynb)



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


