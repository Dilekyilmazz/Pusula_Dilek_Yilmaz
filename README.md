# Pusula Internship Data Science Case

**Name:** Dilek Yılmaz  
**Email:** dilekyilmaz000@gmail.com  

---

## 📌 Project Overview
This project was developed as part of the **Pusula Data Science Internship Case Study**, using the dataset **`Talent_Academy_Case_DT_2025.xlsx`**.  
The goal is to perform **Exploratory Data Analysis (EDA)** and **data preprocessing** steps to clean the raw dataset, handle missing values, transform categorical and multi-label variables, and prepare the data for machine learning models.

### Main steps performed
- Analysis and imputation of missing values  
- Conversion of the string target variable into numeric format  
- Encoding of categorical variables using **OneHotEncoder**  
- Expansion of multi-label columns (`KronikHastalik`, `Alerji`, `Tanilar`, `UygulamaYerleri`) into dummy variables  
- Scaling of numerical variables (**StandardScaler**)  
- Saving the cleaned dataset in multiple formats (`.parquet`, `.csv`, `.xlsx`)  
- Generating **EDA summary CSVs** (target distribution and top 20 values per multi-label column)  

Detailed EDA findings and preprocessing decisions are included in the full report:  
➡️ [EDA and Preprocess Report](reports/EDA_and_Preprocess.md)

---

## ⚙️ How to Run

### Requirements
Python 3.10+ and the following libraries:
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- openpyxl (for reading Excel files)  
- joblib (for saving the pipeline)  

---

### 1. Environment Setup
Create a virtual environment and install the requirements:

```bash
python -m venv .venv
.\.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux / Mac

pip install -r requirements.txt

### 2. Run Preprocessing

# Clean the dataset and save as a single file (Parquet, CSV, Excel)
python src/preprocess.py --data data/raw/Talent_Academy_Case_DT_2025.xlsx --out artifacts/clean_dataset.parquet

# Run with train/test split (optional)
python src/preprocess.py --data data/raw/Talent_Academy_Case_DT_2025.xlsx --train-test

Generated files

artifacts/clean_dataset.parquet → Clean dataset (Parquet format)

artifacts/clean_dataset.csv → Clean dataset (CSV format)

artifacts/clean_dataset.xlsx → Clean dataset (Excel format)

artifacts/preprocessor.joblib → Saved preprocessing pipeline

artifacts/X_train.parquet, artifacts/X_test.parquet, artifacts/y_train.parquet, artifacts/y_test.parquet → Train/Test split (optional)

artifacts/eda_outputs/ → CSV outputs of EDA summaries

### 3. EDA Notebook

To explore EDA and visualizations, run:
jupyter notebook notebooks/clean_dataset_eda.ipynb
This notebook contains:

Target distribution plots

Numerical variable distributions and correlations

Categorical variable frequency plots

Multi-label variable frequency plots

Validation of preprocessing outputs
