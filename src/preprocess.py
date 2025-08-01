import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop customerID if it exists
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Convert TotalCharges to numeric (it may have spaces/NaNs)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    # Encode target column
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Encode categorical features
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df
