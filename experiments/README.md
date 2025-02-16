# 🔬 Model Experiments  

This folder contains detailed experiment results for optimising the **Self-Driving Car AI** model.  
We tested multiple architectures and training strategies to achieve **higher accuracy, better generalisation, and efficient inference**.

---

## 📊 Experiment Summary  

We conducted extensive testing using different architectures and training techniques. Below is a comparison of key models:

| Model                                      | Training Accuracy | Validation Accuracy | MSE    | Techniques Used |
|--------------------------------------------|------------------|------------------|-------|----------------------|
| **EfficientNetB5 + B7 (Ensemble, K-Fold + Aug)** | **98.34%** | **97.32%** | **0.0131** | **Ensembling, K-Fold, Data Augmentation, LR Scheduling** |
| **MobileNetV2 (K-Fold Validation)**        | 97.77% | 97.25% | 0.0134 | K-Fold, Augmentation, Data Preprocessing |
| **EfficientNetB7 (Baseline Training)**     | 97.90% | 97.00% | 0.0138 | Baseline Training, Custom Weight Initialisation |
| **EfficientNetB2 (K-Fold + Augmentation)** | 96.80% | 96.50% | 0.0157 | K-Fold, Augmentation, Label Smoothing |
| **EfficientNetB2 (Image Augmentation, Fixed Epochs)** | 96.40% | 96.14% | 0.0163 | Augmentation (Fixed Epochs), Adaptive LR |
| **EfficientNetB2 (Image Augmentation, Dynamic Epochs)** | 95.50% | 94.75% | 0.0185 | Augmentation (Dynamic Epochs), Early Stopping |
| **MobileNetV2 (Standard Training)**        | 94.50% | 93.90% | 0.0201 | Standard Training, Dropout Regularisation |
| **Baseline Model (Keras Tuner Optimised)** | 91.80% | 90.70% | 0.0213 | Keras Tuner Hyperparameter Tuning, Grid Search |
| **Baseline Neural Network (Simple NN)**    | 89.20% | 88.90% | 0.0247 | Simple NN (Baseline), Batch Normalisation |

---

## 🔑 Key Findings  

📌 **Best Overall Model:**  
- **EfficientNetB5 + B7 (Ensemble, K-Fold, Augmentation)** achieved the best **validation accuracy (97.32%)** and the lowest **MSE (0.0131)**.
- **Ensembling** multiple models significantly improved prediction stability.  

📌 **Techniques That Worked Well:**  
- **K-Fold Cross-Validation** reduced overfitting and improved generalisation.  
- **Data Augmentation** (brightness, contrast, slight rotation) enhanced model robustness.  
- **Custom Learning Rate Scheduling** helped accelerate convergence and avoid overfitting.  

📌 **Performance vs. Efficiency:**  
- **EfficientNetB2** provided a **good trade-off** between accuracy and model size.  
- **MobileNetV2** is a viable choice for **fast inference**, but with a slight accuracy trade-off.  

---

## 🚀 TFLite Deployment  

To optimise the model for **real-world deployment**, we converted a trained EfficientNetB2 model to **TensorFlow Lite (TFLite)** format.  
This enables **faster inference on edge devices** while maintaining competitive accuracy.

### ✅ **Key Benefits of TFLite**
- **Model size reduction** → Lower storage & memory usage.  
- **Faster inference** → Suitable for real-time applications.  
- **Preserves key features** → Efficient inference without major accuracy loss.  

📂 **Files Included**
- `convert_to_tflite.ipynb` → Converts the trained model to **TFLite** format.  

---

## 🏆 Conclusion  

Through extensive experimentation, we found that:  

✅ **EfficientNetB5 + B7 (Ensemble, K-Fold + Augmentation)** achieved the highest **validation accuracy (97.32%)** and the lowest **MSE (0.0131)**.  
✅ **K-Fold Cross-Validation** significantly reduced overfitting and improved model generalisation.  
✅ **Data Augmentation** (brightness, contrast, small rotation) helped stabilise training and enhanced model robustness.  
✅ **MobileNetV2** is a viable alternative for edge deployment due to its faster inference speed, though at a slight accuracy trade-off.  
✅ **TFLite conversion** successfully reduced model size and improved inference speed, making it suitable for real-time self-driving applications.  
✅ **For real-world deployment on embedded self-driving systems, MobileNetV2 + TFLite was selected** due to its smaller footprint and efficient inference capabilities.  

📂 **For full details, check the individual experiment notebooks in this folder.**  
