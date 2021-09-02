import os
import sys
import azureml.core
from pyspark.sql import SparkSession
from azureml.core import Run, Dataset

print(azureml.core.VERSION)
print(os.environ)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file_input")
parser.add_argument("--output_dir")
args = parser.parse_args()

spark = (
    SparkSession
    .builder
    .appName("AML Dataprep")
    .config("spark.executor.cores",1)
    .config("spark.executor.instances", 1)
    .config("spark.executor.memory","1g")
    .config("spark.executor.cores",1)
    .config("spark.executor.instances", 1 )
    .getOrCreate())
sdf = spark.read.option("header", "true").csv(args.file_input)
sdf.show()

sdf.coalesce(1).write\
.option("header", "true")\
.mode("append")\
.csv(args.output_dir)