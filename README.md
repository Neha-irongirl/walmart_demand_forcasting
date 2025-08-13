#  Demand Forecasting for Walmart

##  Overview
This project focuses on **predicting product demand** in a retail/e-commerce environment using advanced **machine learning regression techniques**.  
The goal is to **optimize inventory levels**, **reduce overstock/stockouts**, and **increase profitability** by forecasting demand accurately and integrating predictions into a **reorder recommendation system**.

**Business Problem:**  
Retailers face challenges in balancing stock availability with storage and holding costs.  
An accurate demand forecasting model can:
- Prevent revenue loss from stockouts
- Reduce costs from overstocking
- Enable smarter promotions and marketing decisions

---

##  Objectives
1. **Forecast future demand** for products using historical sales data.
2. Identify **seasonal patterns**, **holiday effects**, and **promotion impacts**.
3. Build a **reorder recommendation engine** that suggests optimal stock replenishment.
4. Visualize demand trends and inventory status for better decision-making.

---

##  Dataset
**Source:**  
- [Walmart Sales Forecasting - Kaggle](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)  
- Alternatively: Web-scraped product sales data or other open datasets.

**Key Features:**
- `store_id` — Unique store identifier
- `item_id` — Product identifier
- `date` — Historical sales date
- `sales` — Number of units sold
- `on_promotion` — Whether the product was on promotion
- `holiday_flag` — Whether the day was a holiday
- `stock_level` — Inventory available (if applicable)

---

##  Tech Stack
- **Programming Language:** Python 3.10+
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning Models:**  
  - Random Forest Regression  
  - XGBoost  
  - LightGBM  
- **Model Evaluation Metrics:** RMSE, MAE, MAPE
- **Deployment:** Streamlit (for interactive dashboard)
- **Version Control:** Git & GitHub

---

##  Approach

### **1. Data Preprocessing**
- Handling missing values
- Converting date columns to `datetime` format
- Feature engineering:
  - Extracting `year`, `month`, `day`, `day_of_week`
  - Flagging weekends
  - Creating **holiday & promotion features**
  - Lag features (sales in previous days/weeks)
  - Rolling mean features for demand smoothing
- One-hot encoding categorical variables

### **2. Exploratory Data Analysis (EDA)**
- Trend analysis of sales over time
- Seasonal demand patterns
- Correlation between promotions & sales spikes
- Stock level vs sales analysis

### **3. Modeling**
- Baseline: Mean & Naïve forecasts
- Machine Learning:
  - Random Forest Regressor
  - XGBoost Regressor
  - LightGBM Regressor
- Hyperparameter tuning using GridSearchCV / Optuna
- Cross-validation to avoid overfitting

### **4. Evaluation**
- Compare RMSE, MAE, and MAPE for all models
- Feature importance analysis
- Error analysis (days with large prediction errors)

### **5. Reorder Recommendation System**
- Use forecasted demand to calculate **reorder points** based on:
  - Lead time
  - Safety stock
- Generate a report with:
  - Product ID
  - Predicted demand
  - Recommended reorder quantity

### **6. Visualization & Dashboard**
- Time series plots of actual vs predicted demand
- Seasonal decomposition plots
- Inventory status vs predicted demand
- Streamlit app for:
  - Selecting store/product
  - Viewing demand forecast
  - Downloading reorder report

---

##  Results
- Achieved **X% improvement** in RMSE compared to baseline
- Identified top sales drivers: holiday promotions, seasonal effects, and product category
- Reduced potential stockouts by **Y%** in simulation

---

##  How to Run Locally

### **1. Clone Repository**
```bash
git clone https://github.com/Neha-irongirl/walmart_demand_forcasting
cd walmart_demand_forcasting
````

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Download Dataset**

* Place the dataset in the `data/` folder.
* Ensure the CSV file is named `sales_data.csv` (or update the config in `config.yaml`).

### **4. Train Model**

```bash
python train.py
```

### **5. Run Streamlit Dashboard**

```bash
streamlit run app.py
```

---

## Repository Structure

```
walmart_demand_forcasting/
├── data/
├── notebooks/
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation.py
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

---

## Future Improvements

* Integrate deep learning models (LSTM, GRU) for better time-series forecasting
* Automate data scraping from e-commerce sites
* Add real-time demand prediction API
* Include cost-benefit analysis of reorder recommendations

---
