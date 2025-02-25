# Explainable Temporal Inference for Irregular Multivariate Time Series. A Case Study for Early Prediction of Multidrug Resistance

Óscar Escudero-Arnanz, Cristina Soguero-Ruiz, Joaquín Álvarez-Rodríguez, Antonio G. Marques

## Abstract
**Objective:** Many real problems in healthcare involve Multivariate Time Series (MTS) as input and Time Series (TS) as output. A key example is predicting Multidrug Resistance (MDR) acquisition in Intensive Care Unit (ICU) patients over time. To enhance interpretability, we propose novel eXplainable Artificial Intelligence (XAI) methods for "MTS-to-TS" inference architectures, enabling time-resolved risk assessments critical for clinical decision-making.
**Methods:** We introduce XAI techniques for "MTS-to-TS" inference, including i) Irregular Time SHapley Additive exPlanation (IT-SHAP), a post-hoc method extending TimeSHAP to TS outputs for time-resolved feature importance; ii) Hadamard Attention, an intrinsic mechanism capturing key temporal dependencies; and iii) Causal Conditional Mutual Information-based feature selection, a pre-hoc approach identifying informative variables before training.
**Results:** We evaluate our approach using 16 years of ICU data from the University Hospital of Fuenlabrada in Spain, covering 71 variables and more than 3,000 admissions. IT-SHAP identifies critical MDR risk factors, such as early antibiotic administration (PEN, SUL) and bacterial cultures (Staphylococcus, Pseudomonas), validated by clinical experts. We also demonstrate generalizability in circulatory failure prediction.
**Conclusion:** Our XAI framework improves interpretability in ``MTS-to-TS'' predictions, with IT-SHAP proving most effective, especially with architectures using attention mechanisms. It enhances explainability in clinical decision-making, identifying key MDR risk factors and generalizing to other ICU conditions.
**Significance:** By integrating real-time, explainable MDR risk predictions into Electronic Health Record systems, our approach enables timely interventions, improved antimicrobial stewardship, and better infection control strategies. Its scalability to other ICU conditions highlights its potential for broader clinical adoption.

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
