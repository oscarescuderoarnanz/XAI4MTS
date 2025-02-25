# MDR - Model Interpretability and Deep Learning Evaluation

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

### **SINGLE_OUTPUT**
Contains a GRU model returning a single prediction for the entire time series. This is used to evaluate interpretability with the **Timeshap** method located in `timeshap/`.

### **`timeshap/`**
Contains the implementation of the methodology proposed in [3], applied to the MDR dataset:  
> [3] Bento, J., Saleiro, P., Cruz, A. F., Figueiredo, M. A., & Bizarro, P. (2021, August). Timeshap: Explaining recurrent models through sequence perturbations. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining (pp. 2565-2573).

## **Prerequisites**
- Required libraries and dependencies are listed in `..\requirements.txt`.
