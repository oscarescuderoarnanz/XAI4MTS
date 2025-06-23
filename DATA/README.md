## **Directory Structure**

### **`/MDR`, `/CIRCULATORY`
Each of these folders contains:
- **`s1`, `s2`, `s3`:** These are different train, test, and validation partitions created from the respective datasets.
- **`FS_CCMI`:** This folder contains partitions (`s1`, `s2`, `s3`) created using the features selected by Conditional Mutual Information (CMI).

### **`/PREPROCESSING`**
This folder includes the preprocessing scripts


#### **CIRCULATORY**
- **`Step1_HIRID_Preprocessing_T=8`:** Preprocessing script for the circulatory failure dataset.
- **`Step2_GeneratePartitions`:** Script to generate train, test, and validation partitions.

---

## **Data Availability**

### **MDR**
- The data used for MDR experiments is **private** and cannot be uploaded to this repository.

### **CIRCULATORY**
- The raw data must be downloaded by the user from the official [PhysioNet's website](https://physionet.org/content/hirid/1.1.1/) after fulfilling specific requirements. Visit the official PhysioNet page for more details on accessing the data. The annotation of the dataset is associated with the paper:
    >*Hyland, S.L., Faltys, M., Hüser, M. et al. Early prediction of circulatory failure in the intensive care unit using machine learning. Nat Med 26, 364–373 (2020)*.


