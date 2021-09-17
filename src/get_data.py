import os
import wget
import zipfile


if __name__ == "__main__":
    # Download the zipped dataset
    url = 'MY-DATASET-URL-AWS'  # TODO: Upload the CSV to S3 and get the link
    zip_name = "data.zip"
    wget.download(url, zip_name)

    # Unzip it and standardize the .csv filename
    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        zip_ref.filelist[0].filename = 'data/creditcard.csv'
        zip_ref.extract(zip_ref.filelist[0])

    os.remove(zip_name)
