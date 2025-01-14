# RNN Architectures

This directory contains the implementation of various Recurrent Neural Network (RNN) architectures and supporting modules for the experiments. These architectures are designed for tasks involving irregular multivariate time series.

### **`pos_hoc.py`**
This file contains the implementations of the following RNN architectures:
- **GRU (Gated Recurrent Unit)**  
- **LSTM (Long Short-Term Memory)**  
- **Vanilla RNN**  

The specific RNN architecture executed depends on the `model_type` variable chosen during the experiments. This allows seamless experimentation with different temporal models.

---

### **`intrinsec.py`**
Similar to `pos_hoc.py`, this file implements the same RNN architectures (GRU, LSTM, Vanilla RNN). However, it incorporates the **attention mechanism** defined in `att_method.py` into the models, enabling intrinsic interpretability by highlighting important features and time steps during inference.

---

### **`pre_hoc.py`**
This file is dedicated to **Conditional Mutual Information (CMI)**, which is used as a pre-hoc explainability method. CMI selects the most relevant features before training the model, enhancing model performance and interpretability by focusing on key predictors.

---

### **`att_method.py`**

This file contains the implementation of the **attention mechanism**. The attention mechanism is integrated into the architectures in `intrinsec.py`

---

### **`utils.py`**
This file includes Temporal Balance Binary Cross Entropy, a custom loss function tailored for imbalanced binary classification tasks over temporal data

---