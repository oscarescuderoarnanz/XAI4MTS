# Explainable Temporal Inference for Irregular Multivariate Time Series. A Case Study for Early Prediction of Multidrug Resistance

Óscar Escudero-Arnanz, Cristina Soguero-Ruiz, Joaquín Álvarez-Rodríguez, Antonio G. Marques

## Abstract
**Objective:** Many healthcare problems involve complex patient trajectories represented as Multivariate Time Series (MTS), with predictions often coming as Time Series (TS) outputs. Despite recent advances, these "MTS-to-TS" inference tasks remain challenging due to data irregularity, temporal dependencies, and the need for clinical explainability. To address these demands, we propose novel eXplainable Artificial Intelligence (XAI) methods for "MTS-to-TS" architectures, enabling tracking of patient evolution and identification of key variable patterns associated with adverse outcomes. We evaluate our approach on private ICU data from the University Hospital of Fuenlabrada (UHF) for Multidrug Resistance (MDR) prediction and the public HiRID dataset (circulatory failure).
**Methods:** We introduce three XAI techniques: i) Irregular Time SHapley Additive exPlanation (IT-SHAP), a post-hoc extension of TimeSHAP to TS outputs; ii) Hadamard Attention, an intrinsic mechanism for capturing temporal dependencies; and iii) Causal Conditional Mutual Information, a pre-hoc approach for feature selection.
**Results:** MDR prediction achieved highest performance with a GRU using Hadamard Attention (ROC-AUC=0.783 ± 0.023), while circulatory failure was best predicted with LSTM (ROC--AUC of 0.9970 ± 1.6e^-3). In terms of explainability, IT-SHAP uncovered clinically relevant risk factors—early antibiotic use and bacterial cultures—later validated by UHF clinicians.
**Conclusion:** Our framework offers temporal explainability in ``MTS-to-TS'' architectures, allowing clinicians to trace disease trajectories and understand the contribution of each variable at each time step.
**Significance:** Integrating explainable MDR risk predictions into EHR systems enables early interventions, improved antimicrobial stewardship, and infection control. The framework’s scalability to other ICU challenges underscores its clinical impact.

> Submission Status: This manuscript has been submitted to IEEE Transactions on Biomedical Engineering and is currently under review.

## Project Structure

The repository is organized into the following main directories and files:

### **`DATA/`**
Contains the data used for the experiments, organized into subfolders for each dataset. The datasets consist of irregular multivariate time series, where the length of the time series varies across instances. The project focuses on binary classification tasks based on these temporal data. Further details are provided in the README files within each subfolder.

- **`MDR`**:  
  The MDR dataset is collected from the University Hospital of Fuenlabrada (Madrid, Spain). Due to data protection regulations, this dataset is private and cannot be shared publicly.

- **`CIRCULATORY`**:  
  The imputed dataset is part of the HiRID collection and is publicly available but requires fulfilling certain access requirements. You can request access via [PhysioNet's website](https://physionet.org/content/hirid/1.1.1/). The dataset annotation is associated with [1]:  
  > [1] *Hyland, S.L., Faltys, M., Hüser, M. et al. Early prediction of circulatory failure in the intensive care unit using machine learning. Nat Med 26, 364–373 (2020)*.

---
### **`src/`**

- #### **`code/`**

  This directory contains all the main scripts and implementations for the project. Further details about each submodule are available in the respective README files.

    - **`explainability_methods/`**
        - **`IT_SHAP/`**: Implementation of the IT-SHAP methodology

        - **`att_method.py`**: Script implementing Hadamard attention mechanism for model interpretability
  
        - **`pre_hoc.py`**: Script for pre-hoc explainability using Causal Conditional Mutual Information

    - **`rnns_architectures/`**
        - **`pos_hoc.py`**: Recurrent Neural Networks (RNN), including Vanilla RNN, GRU, and LSTM
          
        - **`intrinsec.py`**: RNN-based models with Hadamard attention mechanism

    - **`non_rnns_architectures/`**
        - **`transformer_poshoc.py`**: Transformer 
          
        - **`transformer_intrinsec.py`**: Transformer model with Hadamard attention mechanism 

    - **`utils.py`**: Utility functions, including the Temporal Balance Binary Cross Entropy loss function for handling imbalance


- #### **`experiments/`**
    This folder contains all the experiments conducted on the datasets. Each dataset has its subfolder with its own README file for more details:
      
    - **`CIRCULATORY/`**: Experiments using circulatory failure data
 
      
    - **`MDR/`**: Experiments related to predicting MDR

- #### **`results/`**
    This folder stores the main inference and interpretability results from all experiments
---

### **`requirements.txt`**
File containing the dependencies required to run this project. It is recommended to install these dependencies in a virtual environment to avoid conflicts
