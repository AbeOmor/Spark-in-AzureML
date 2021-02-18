'''
STEP 3: Process data using Spark

Note - This requires the spark-xml maven library (com.databricks:spark-xml_2.11:0.6.0) to be installed.
'''
import pyspark
from pyspark.sql import functions as sf
from pyspark.sql import Row
from pyspark.sql.functions import size, col, concat_ws, rtrim, regexp_replace, split, udf
from pyspark.sql.types import ArrayType

# load xml file into spark data frame.
posts = spark.read.format("xml").option("rowTag", "row").load("dbfs:/tmp/posts/Posts.xml")

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