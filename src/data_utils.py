import boto3
import botocore


def download_data_from_s3(bucket_name, key, dst):
    try:
        s3 = boto3.resource('s3')
        s3.Bucket(bucket_name).download_file(key, dst)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise


def upload_data_to_s3(bucket_name, filepath, key_dst):
    try:
        s3 = boto3.resource('s3')
        s3.Bucket(bucket_name).upload_file(filepath, key_dst)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("Upload not successful.")
        else:
            raise
