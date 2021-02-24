#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import numpy as np
import pyspark
import os
import urllib
import sys
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os

from pyspark.sql.functions import *
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType, IntegerType, StringType

from azureml.core.run import Run

# initialize logger
run = Run.get_context()

# start Spark session
spark = pyspark.sql.SparkSession.builder.getOrCreate()
print("Spark session created")

# print runtime versions

print('****************')
print('Python version: {}'.format(sys.version))
print('Spark version: {}'.format(spark.version))
print(os.environ)
print('****************')

'''
STEP 1: Download Stack Overflow data from archive. (This takes about 2-3 hours)
'''

# Define input arguments
import sys
data_dir = sys.argv[1]

xml_file_path = str(os.path.join(data_dir, 'stackoverflow/posts/Posts.xml'))

print('*******TESTING PATHS FOR DATA*********')
print(xml_file_path)
print(os.listdir(str(os.path.join(data_dir, 'stackoverflow'))))
print('****************')


'''
STEP 2: Process data using Spark

Note - This requires the spark-xml maven library (com.databricks:spark-xml_2.11:0.6.0) to be installed.
'''
import pyspark
from pyspark.sql import functions as sf
from pyspark.sql import Row
from pyspark.sql.functions import size, col, concat_ws, rtrim, regexp_replace, split, udf
from pyspark.sql.types import ArrayType

# load xml file into spark data frame.
posts = spark.read.format("xml").option("rowTag", "row").load(xml_file_path)

# select only questions
questions = posts.filter(posts._PostTypeId == 1) 

# drop irrelvant columns and clean up strings
questions = questions.select([c for c in questions.columns if c in ['_Id','_Title','_Body','_Tags']])
questions = questions.withColumn('full_question', sf.concat(sf.col('_Title'), sf.lit(' '), sf.col('_Body')))
questions = questions.select([c for c in questions.columns if c in ['_Id','full_question','_Tags']]).withColumn("full_question", regexp_replace("full_question", "[\\n,]", " "))
questions = questions.withColumn("_Tags", regexp_replace("_Tags", "><", " "))
questions = questions.withColumn("_Tags", regexp_replace("_Tags", "(>|<)", ""))
questions = questions.withColumn('_Tags', rtrim(questions._Tags))
questions = questions.withColumn('_Tags', split(questions._Tags, " "))

# filter out to single tags in following list
tags_of_interest = ['azure-devops', 'azure-functions', 'azure-web-app-service', 'azure-storage', 'azure-virtual-machine'] 

def intersect(xs):
    xs = set(xs)
    @udf("array<string>")
    def _(ys):
        return list(xs.intersection(ys))
    return _

questions = questions.withColumn("intersect", intersect(tags_of_interest)("_Tags"))
questions = questions.filter(size(col("intersect"))==1)
questions = questions.select('_Id', 'full_question', 'intersect').withColumn('_Tags', concat_ws(', ', 'intersect'))
questions = questions.select('_Id', 'full_question', '_Tags')

questions.show()


'''
Step 3: Convert processed data into pandas data frame for final preprocessing and data split
'''

df = questions.toPandas()

# drop nan values and remove line breaks
df.dropna(inplace=True)
df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)

# balance dataset 
balanced = df.groupby('_Tags')
balanced.apply(lambda x: x.sample(balanced.size().min())).reset_index(drop=True).to_csv('balanced.csv')
bd = pd.read_csv('balanced.csv')
bd.drop('Unnamed: 0', axis=1, inplace=True)

# shuffle data 
# bd = shuffle(bd)

# split data into train, test, and valid sets
msk = np.random.rand(len(bd)) < 0.7
train = bd[msk]
temp = bd[~msk]
msk = np.random.rand(len(temp)) < 0.66
valid = temp[msk]
test = temp[~msk]

'''
STEP 4: Save dataset into csv and class.txt files
'''

output_dir = str(os.path.join(data_dir, 'stackoverflow/output2'))


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# create and save classes.txt file
classes = pd.DataFrame(bd['_Tags'].unique().tolist())
classes.to_csv(os.path.join(output_dir, 'classes.txt'), header=False, index=False)

# save train, valid, and test files
train.to_csv(os.path.join(output_dir, 'train.csv'), header=False, index=False)
valid.to_csv(os.path.join(output_dir, 'valid.csv'), header=False, index=False)
test.to_csv(os.path.join(output_dir, 'test.csv'), header=False, index=False)
