# MLOps: A Complete and Hands-on Introduction - Part 2

In the first part of this series we could explore the idea of MLOps, some of its frameworks and other tools that can be useful when we want to apply this concept in our daily bases. Although we saw some tools there, I think it deserves some hands-on examples on how it can be used on a real data science project.

The dataset we'll be using in this hands-on article is available on Kaggle, and it's called [Credit Card Fraud Detection][4]. Moreover, we added the data processing step based on this [notebook: Credit Fraud || Dealing with Imbalanced Datasets][5], that is available on Kaggle as well.

Here we will be using GitHub for hosting our repository, [Data Version Control][1] (DVC) for managing and reproducing the data science pipeline we are going to build. In addiction, we are going to store within a S3 bucket all the data that are too big to be tracked by Git, such as our models and the dataset. 

Following the MLOps principles, we are going to use [GitHub Actions][2] to automate all our development workflow, and we will integrate the repository with [Heroku][3] in order to automatically deploy our API and serve our model. With that being covered, in the next few sections we will dive into this very simple MLOps pipeline and see how can we implement the concepts we learnt before.

## Download the dataset

First things first, we can't do anything without being able to access our dataset. So, we have to develop a routine that can help us in downloading the data from our S3 bucket.
The following code snippet do the trick for us, the method takes as parameters the name of your bucket (bucket_name), the object key (key), and the destination (dst) where you'd like to store the data locally.

```python
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
 ```

To make our lives easier, we've created a script called ```download_dataset.py``` to be used as the first step of our pipeline managed by DVC. 
In this script, we are just defining the information we need to use the ```download_data_from_s3``` method in order to download the dataset.

## Data Preparation
## Model Training and Validation
## Model Deploying and Serving


[1]: https://dvc.org/
[2]: https://github.com/features/actions
[3]: https://dashboard.heroku.com/apps
[4]: https://www.kaggle.com/mlg-ulb/creditcardfraud
[5]: https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets
