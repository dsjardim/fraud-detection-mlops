import boto3
import botocore

if __name__ == "__main__":
    BUCKET_NAME = 'credit-fraud-dataset'  # replace with your bucket name
    KEY = 'creditcard.csv'  # replace with your object key
    s3 = boto3.resource('s3')

    try:
        s3.Bucket(BUCKET_NAME).download_file(KEY, 'data/creditcard.csv')
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
