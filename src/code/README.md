# RNN Architectures and Explainability Methods

This directory contains the implementation of various Recurrent Neural Network (RNN) architectures and explainability methods tailored for experiments involving irregular multivariate time series. It also includes supporting modules to facilitate experimentation, interpretability, and model evaluation.

---

### **`explainability_methods/`**

- #### **`IT_SHAP/`**
    Contains the implementation of the **IT-SHAP** methodology, which provides explainable temporal inference for irregular multivariate time series:
    - Computing feature importance for each time step, offering granular insights into model predictions
    - Operating directly on irregular multivariate time series without requiring imputation or regularization
    - 
- #### **`att_method.py`**
    - Implements a **Hadamard attention mechanism** that identifies important features and time steps. This module is integrated into certain RNN architectures, such as those in `intrinsec.py`, to enhance intrinsic interpretability.

- #### **`pre_hoc.py`**
    - Dedicated to **Causal Conditional Mutual Information (CCMI)** as a pre-hoc explainability method

---

### **`rnns_architectures/`**

This subdirectory contains the implementations of various RNN-based models used in the experiments

- #### **`pos_hoc.py`**
    Implements core RNN architectures:
    - **GRU (Gated Recurrent Unit)**  
    - **LSTM (Long Short-Term Memory)**  
    - **Vanilla RNN**  

    *The specific architecture used during experiments is determined by the `model_type` variable. These models are designed to work with irregular temporal data, providing flexibility for different experimental setups*

- #### **`intrinsec.py`**
    Builds on the RNN architectures in `post_hoc.py` by incorporating the **Hadamard attention mechanism** (defined in `att_method.py`). This addition highlights important features and time steps during inference, offering intrinsic interpretability for temporal data

---

### **`utils.py`**
A utility module that includes:
- **Temporal Balance Binary Cross Entropy**: A custom loss function designed for imbalanced binary classification tasks over temporal data.
- Additional helper functions
