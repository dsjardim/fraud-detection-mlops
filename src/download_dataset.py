import os
from data_utils import download_data_from_s3

if __name__ == "__main__":
    BUCKET_NAME = 'credit-fraud-mlops-artifacts'  # replace with your bucket name
    KEY = 'creditcard.csv'  # replace with your object key
    os.mkdir('data') if not os.path.exists('data') else None

    download_data_from_s3(BUCKET_NAME, KEY, 'data/creditcard.csv')
