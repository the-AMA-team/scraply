import sagemaker
from sagemaker.pytorch import PyTorch
import os
import boto3

boto3.setup_default_session(region_name='us-east-1')

sts = boto3.client('sts')
print("AWS Identity:", sts.get_caller_identity()["Arn"])
# print(s3.list_objects_v2(Bucket='sagebucket4', Prefix='datasets/'))

# role = "arn:aws:iam::842675993918:role/service-role/AmazonSageMaker-ExecutionRole-20250222T132005"
# role = "arn:aws:iam::842675993918:role/SageRole"
role = sagemaker.get_execution_role()
bucket = "sagemaker-us-east-1-842675993918"
s3_uri = f"s3://{bucket}/datasets/"

# os.system("dir")


estimator = PyTorch(
    entry_point="train.py",
    role=role,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    framework_version="1.12",
    py_version="py38",
    hyperparameters={
        "epochs": 2,
        "batch-size": 8,
        "learning-rate": 0.001,
    },
    enable_cloudwatch_metrics=True,
    log_level="DEBUG",
    output_path=f"s3://{bucket}/models/"
    # Define input channel
)

try:
    estimator.fit({'train': s3_uri},wait=False)
    print(f"✅ Job submitted: {estimator.latest_training_job_name}")
except Exception as e:
    print(f"❌ Submission failed: {e}")
    raise  # Show full traceback

predictor = estimator.deploy(instance_type="ml.g4dn.xlarge", initial_instance_count=1)
response = predictor.predict(["Alice was sleepy"])
print(response)
