# 🔬 Model Experiments  

This folder contains detailed experiment results for optimising the **Self-Driving Car AI** model.  
We tested multiple architectures and training strategies to achieve **higher accuracy, better generalisation, and efficient inference**.

---

## 📊 Experiment Summary  

We conducted extensive testing using different architectures and training techniques. Below is a comparison of key models:

| Model                                 | Training Accuracy | Validation Accuracy | MSE    | Techniques Used |
|---------------------------------------|------------------|------------------|-------|----------------------|
| **EfficientNetB5 + B7 (Ensemble, K-Fold + Aug)** | **98.34%** | **97.32%** | **0.0131** | **Ensembling, K-Fold, Data Augmentation, LR Scheduling** |
| **MobileNetV2 (K-Fold Validation)** | 97.77% | 97.25% | 0.0134 | K-Fold, Augmentation, Data Preprocessing |
| **EfficientNetB7 (Baseline Training)** | 97.90% | 97.00% | 0.0138 | Baseline Training, Custom Weight Initialisation |
| **EfficientNetB2 (K-Fold + Augmentation)** | 96.80% | 96.50% | 0.0157 | K-Fold, Augmentation, Label Smoothing |
| **EfficientNetB2 (Image Augmentation, Fixed Epochs)** | 96.40% | 96.14% | 0.0163 | Augmentation (Fixed Epochs), Adaptive LR |
| **MobileNetV2 (Standard Training)** | 94.50% | 93.90% | 0.0201 | Standard Training, Dropout Regularisation |
| **Baseline Neural Network (Simple NN)** | 89.20% | 88.90% | 0.0247 | Basic MLP, Batch Normalisation |

---

## 🔑 Key Findings  

📌 **Best Overall Model:**  
- **EfficientNetB5 + B7 (Ensemble, K-Fold, Augmentation)** achieved the best **validation accuracy (97.32%)** and the lowest **MSE (0.0131)**.
- **Ensembling** multiple models significantly improved prediction stability.  

📌 **Techniques That Worked Well:**  
- **K-Fold Cross-Validation** reduced overfitting and improved generalisation.  
- **Data Augmentation** (brightness, contrast, slight rotation) enhanced model robustness.  
- **Custom Learning Rate Scheduling** helped accelerate convergence.  

📌 **Performance vs. Efficiency:**  
- **EfficientNetB2** provided a **good trade-off** between accuracy and model size.  
- **MobileNetV2** is a viable choice for **fast inference**, but with a slight accuracy trade-off.  

---

## 🏆 Conclusion  

Through extensive experimentation, we found that:  

✅ **EfficientNetB5 + B7 (Ensemble, K-Fold + Augmentation)** achieved the highest **validation accuracy (97.32%)** and the lowest **MSE (0.0131)**.  
✅ **K-Fold Cross-Validation** significantly reduced overfitting and improved model generalisation.  
✅ **Data Augmentation** (brightness, contrast, small rotation) helped stabilise training and enhanced model robustness.  
✅ **MobileNetV2** is a viable alternative for edge deployment due to its faster inference speed, though at a slight accuracy trade-off.  

📌 **Future Improvements:**  
- 🏗 **Explore Transformer-based models** (e.g., Vision Transformers, Swin Transformers).  
- 📉 **Further optimise inference latency** for real-world deployment using TensorFlow Lite & model quantisation.  
- 🧠 **Investigate self-supervised learning approaches** for better data efficiency.  

📂 **For full details, check the individual experiment notebooks in this folder.**  
