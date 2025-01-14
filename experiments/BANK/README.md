# BANKRUPTCY - Model Interpretability and Deep Learning Evaluation

This repository implements and analyzes various techniques for interpretability and performance evaluation of deep learning models, with a focus on Recurrent Neural Networks (RNNs) and temporal outputs. It includes implementations for GRU, LSTM, and Vanilla RNN models, as well as methods for evaluating model interpretability and performance.

---

## Directory Structure

### **Figures**
Contains all the generated visualizations from the experiments and analyses.

### **POST-HOC**
1. `exec_model_GRU.py`: Script to run GRU model (LSTM and Vanilla RNN also available).
2. `exec_IT-SHAP.py`: Script to evaluate interpretability using the IT-SHAP technique.
3. `analysis_inference_results.ipynb`: Notebook for analyzing the performance results of the RNN models.
4. `test_xai.ipynb`: Notebook for examining interpretability outcomes using IT-SHAP.

### **INTRINSIC**
1. `exec_model_GRU.py`: Script to run GRU model with attention mechanism (LSTM and Vanilla RNN also available).
2. `analysis_inference_results.ipynb`: Notebook for analyzing the performance results of the RNN models.
3. `test_xai.ipynb`: Notebook for examining interpretability outcomes from attention mechanisms.

### **PRE-HOC**
1. `exec_CMI.py`: Script to run Conditional Mutual Information (CMI).
2. `exec_model_GRU.py`: Script to run a GRU model trained using features selected by CMI.
3. `test_xai.ipynb`: Notebook for examining interpretability outcomes from CMI.

## **Prerequisites**
- Required libraries and dependencies are listed in `..\requirements.txt`.
