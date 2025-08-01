import os
import pandas as pd
from src.preprocess import preprocess_data
from src.download import download_telco_dataset
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import classification_report

# Step 1: Download dataset
dataset_path = download_telco_dataset()
data_file = os.path.join(dataset_path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Step 2: Load data
df = pd.read_csv(data_file)
df = preprocess_data(df)

# Continue training...
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'verbosity': -1
}

model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], num_boost_round=100, early_stopping_rounds=10)
model.save_model('../models/lightgbm_churn.txt')

y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)
print(classification_report(y_test, y_pred_binary))
