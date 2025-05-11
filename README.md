# Telecom Customer Churn Prediction

## 📝 Overview

In the telecom industry, customers frequently switch operators if not offered attractive schemes. Preventing existing customers from churning is critical. As a data scientist, your task is to build an ML model to predict if a customer will churn in a particular month, based on historical data.

## 📌 Problem Statement

- Customers can choose from multiple service providers and often switch.
- The industry faces a 15–25% annual churn rate.
- Retaining a customer is 5–10× cheaper than acquiring a new one.
- The business goal: **Predict which customers are at high risk of churn** using customer-level data and ML models.

## 📂 Data Description

**Main Files:**
- `train.csv` - Training data (69,999 rows × 172 cols)
- `test.csv` - Test data (30,000 rows × 171 cols)
- `sample.csv` - Sample submission (30,000 rows × 2 cols)
- `data_dictionary.csv` - Column descriptions (36 rows × 2 cols)

**Sample Submission:**

| id    | churn_probability |
|-------|------------------|
| 69999 | 0                |
| 70000 | 0                |
| ...   | ...              |

**Sample Train Data:**

| id | circle_id | loc_og_t2o_mou | std_og_t2o_mou | ... | churn_probability |
|----|-----------|----------------|---------------|-----|------------------|
| 0  | 109       | 0.0            | 0.0           | ... | 0                |
| 1  | 109       | 0.0            | 0.0           | ... | 0                |

**Acronyms & Abbreviations:**

| Acronym     | Description                                           |
|-------------|-------------------------------------------------------|
| CIRCLE_ID   | Telecom circle area                                   |
| LOC         | Local calls within same circle                        |
| STD         | STD calls outside the circle                          |
| IC          | Incoming calls                                        |
| OG          | Outgoing calls                                        |
| T2T         | Operator T to T (within same operator)                |
| T2M         | Operator T to other operator mobile                   |
| T2O         | Operator T to other operator fixed line               |
| T2F         | Operator T to fixed lines of T                        |
| T2C         | Operator T to its own call center                     |
| ARPU        | Average revenue per user                              |
| MOU         | Minutes of usage (voice calls)                        |
| AON         | Age on network (days)                                 |
| ONNET       | Calls within same operator network                    |
| OFFNET      | Calls outside operator T network                      |
| ROAM        | Roaming zone indicator                                |
| SPL         | Special calls                                         |
| ISD         | International calls                                   |
| RECH        | Recharge                                              |
| NUM         | Number                                                |
| AMT         | Amount (local currency)                               |
| MAX         | Maximum                                               |
| DATA        | Mobile internet                                       |
| 3G/2G       | 3G/2G network                                         |
| AV          | Average                                               |
| VOL         | Mobile internet volume (MB)                           |
| PCK         | Prepaid service packs                                 |
| NIGHT       | Night hour schemes                                    |
| MONTHLY     | Monthly validity schemes                              |
| SACHET      | Short validity schemes                                |
| *.6, *.7, *.8 | KPIs for June, July, August                         |
| FB_USER     | Facebook scheme user                                  |
| VBC         | Volume-based cost                                     |

## 🛠️ Importing Libraries

Data Structures

import pandas as pd
import numpy as np
import re
import os
import missingno as msno
Sklearn & ML

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, precision_score, recall_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
Plotting

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
Other

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.width', 10000)
text

## 📥 Loading Data

data = pd.read_csv('train.csv')
unseen = pd.read_csv('test.csv')
sample = pd.read_csv('sample.csv')
dd = pd.read_csv('data_dictionary.csv')
text

## 🗂️ Project Structure

.
├── data/
│ ├── train.csv
│ ├── test.csv
│ ├── sample.csv
│ └── data_dictionary.csv
├── notebooks/
│ └── analysis.ipynb
├── src/
│ └── model.py
├── README.md
