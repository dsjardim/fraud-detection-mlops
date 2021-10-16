from pathlib import Path

from data_utils import download_data_from_s3

if __name__ == "__main__":
    BUCKET_NAME = 'credit-fraud-dataset'  # replace with your bucket name
    KEY = 'creditcard.csv'  # replace with your object key
    Path("../data").mkdir(parents=True, exist_ok=True)

    download_data_from_s3(BUCKET_NAME, KEY, '../data/creditcard.csv')
