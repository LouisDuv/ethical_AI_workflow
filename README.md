# Ethics of AI – Making Ethical a Classic AI Workflow

This notebook explores how to **embed ethical considerations into a traditional AI workflow**.  
It walks through data collection, preprocessing, bias exploration, and fine-tuning of a lightweight language model.

---

## 📂 Project Overview

The project focuses on:
- **Data ingestion and cleaning** of a loan dataset.
- **Exploration of biases** within the data.
- **Fine-tuning** of a small transformer-based model (`prajjwal1/bert-tiny`) for downstream tasks.

---

## ⚙️ Workflow Outline

### 1. Downloading and Data Ingestion
The notebook starts by importing and cleaning a loan dataset located at `../data/Loan.csv`.

#### Main steps:
- Importing required libraries:
  ```python
  import pandas as pd
  import seaborn as sns
  import numpy as np
  import random
  from matplotlib import pyplot as plt
  from scipy.stats import chi2_contingency
  ```
- Selecting relevant columns such as:
  ```
  Age, MaritalStatus, EmploymentStatus, EducationLevel, Experience,
  MonthlyLoanPayment, MonthlyIncome, UtilityBillsPaymentHistory, LoanApproved
  ```
- Setting a random seed for reproducibility.

---

### 2. Data Cleaning and Exploration
- Basic data wrangling is performed to remove irrelevant or incomplete features.
- Visual exploration uses **matplotlib** and **seaborn** to analyze feature distributions and detect potential imbalances or biases.

---

### 3. Bias Analysis
- The notebook includes a section explicitly dedicated to **bias exploration**.
- It uses statistical tests such as **Chi-squared tests (`chi2_contingency`)** to assess dependencies between categorical variables and the loan approval outcome.

---

### 4. Fine-Tuning a Language Model

#### Configuration:
| Parameter | Description |
|------------|-------------|
| `MAX_LEN` | Maximum token length |
| `EPOCHS` | Number of training epochs |
| `BATCH` | Training batch size |
| `MODEL` | `prajjwal1/bert-tiny` |

#### Rationale:
- The chosen model is small and efficient, providing a **good trade-off between accuracy and resource usage**.
- The notebook specifies that a **full fine-tuning** approach was selected for better performance on the dataset.

#### Data Preparation:
- Drops unnecessary features.
- Splits features and target variable.
- Converts tabular data into a **textual format** suitable for LLM input.
- Splits data into **training (70%)** and **testing (30%)** sets.

#### Tokenization:
- Texts are tokenized before model training to ensure compatibility with the transformer architecture.

---

## 🧠 Key Concepts Highlighted
- Importance of **ethical awareness** in AI model development.
- Examination of **biases in data** prior to model training.
- Use of **transparent fine-tuning methods** on compact models.

---

## 📄 Dependencies
Ensure the following packages are installed before running the notebook:
```bash
uv add pandas numpy seaborn matplotlib scipy transformers torch scikit-learn lime shap
```

---

## 🧩 File Structure
```
notebook.ipynb
└── ../data/
    └── Loan.csv
```

---

## 📊 Outputs
- Visualizations for data exploration and bias detection.
- Fine-tuned model results (details within the notebook).
