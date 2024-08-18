from time import sleep
from prefect_aws import S3Bucket, AwsCredentials
import os

def create_aws_creds_block():

    aws_access_key = os.environ["AWS_ACCESS_KEY"]
    aws_secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]

    my_aws_creds_obj = AwsCredentials(
        # aws_access_key_id="123abc", aws_secret_access_key="abc123"
        aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_access_key
    )
    my_aws_creds_obj.save(name="my-aws-creds", overwrite=True)


def create_s3_bucket_block():
    aws_creds = AwsCredentials.load("my-aws-creds")
    my_s3_bucket_obj = S3Bucket(
        bucket_name="mlops-temperature-prediction", credentials=aws_creds
    )
    my_s3_bucket_obj.save(name="s3-bucket-block", overwrite=True)


if __name__ == "__main__":
    create_aws_creds_block()
    sleep(5)
    create_s3_bucket_block()