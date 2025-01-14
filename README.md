# IT-SHAP: Explainable Temporal Inference for Irregular Multivariate Time Series

Óscar Escudero-Arnanz, Cristina Soguero-Ruiz, Joaquín Álvarez-Rodríguez, Antonio G. Marques

## Abstract

---

## Project Structure

The repository is organized into the following main directories and files:

### **`DATA/`**
Contains the data used for the experiments, organized into subfolders for each dataset. The datasets consist of irregular multivariate time series, where the length of the time series varies across instances. The project focuses on binary classification tasks based on these temporal data.

- **`MDR`**:  
  The multidrug resistance (MDR) dataset is collected from the University Hospital of Fuenlabrada (Madrid, Spain). Due to data protection regulations, this dataset is private and cannot be shared publicly.

- **`CIRCULATORY`**:  
  The imputed dataset is part of the HiRID collection and is publicly available but requires fulfilling certain access requirements. You can request access via [PhysioNet's website](https://physionet.org/content/hirid/1.1.1/). The dataset annotation is associated with [1]:  
  > [1] *Hyland, S.L., Faltys, M., Hüser, M. et al. Early prediction of circulatory failure in the intensive care unit using machine learning. Nat Med 26, 364–373 (2020)*.

- **`BANKRUPTCY`**:  
  The bankruptcy dataset is publicly available. You can obtain it from the [GitHub Repository](https://github.com/sowide/bankruptcy_dataset/tree/main). The dataset is associated with [2]:  
  > [2] *Machine Learning for Bankruptcy Prediction in the American Stock Market: Dataset and Benchmarks* - *Future Internet MDPI 2022*.

---

### **`experiments/`**
This folder contains all the experiments conducted on the datasets. Each dataset has its subfolder with additional README files for more details:
- **`MDR/`**: Experiments related to predicting MDR.
- **`CIRCULATORY/`**: Experiments using circulatory failure data.
- **`BANK/`**: Experiments with the bankruptcy dataset.

---

### **`rnns_architectures/`**
This directory contains the implementations of the RNN architectures and utilities:
- **`pos-hoc.py`**: Recurrent Neural Networks (RNN), including Vanilla RNN, GRU, and LSTM.
- **`intrinsec.py`**: RNN-based models with Attention Mechanism.
- **`pre-hoc.py`**: Script to run pre-hoc explainability (Conditional Mutual Information, CMI).
- **`utils.py`**:  Includes the Temporal Balance Binary Cross Entropy, a custom loss function for handling imbalanced binary classification tasks over temporal data. Refer to the paper for more details.


---

### **`IT_SHAP/`**
Implementation of the proposed **IT-SHAP methodology**, for explainable temporal inference on irregular multivariate time series (MTS). This module includes functions and logic to evaluate model interpretability.

It addresses two key challenges in this domain:
1. **Computation of feature importance for temporal models** (e.g., RNNs):  
   IT-SHAP evaluates models that generate temporal outputs, this is, a vector containing predictions for each time step, providing granular insights into feature contributions across time.
2. **Support for irregular MTS inputs without preprocessing**:  
   Unlike traditional approaches, IT-SHAP operates directly on irregular temporal data, eliminating the need for imputation or regularization steps.  

These capabilities enable IT-SHAP to handle the complexities of irregular temporal data while offering interpretable insights into model predictions.

---

### **`timeshap/`**
Contains the implementation of the methodology proposed in [3], applied to the MDR dataset:  
> [3] Bento, J., Saleiro, P., Cruz, A. F., Figueiredo, M. A., & Bizarro, P. (2021, August). Timeshap: Explaining recurrent models through sequence perturbations. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining (pp. 2565-2573).

---

### **`results/`**
This folder stores the main inference and interpretability results from all experiments.

---

### **`requirements.txt`**
File containing the dependencies required to run this project. It is recommended to install these dependencies in a virtual environment to avoid conflicts.
