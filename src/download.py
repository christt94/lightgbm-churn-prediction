import kagglehub

def download_telco_dataset():
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    print("✅ Dataset downloaded to:", path)
    return path
