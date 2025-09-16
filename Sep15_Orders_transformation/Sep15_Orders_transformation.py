import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import *



## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read validated data
df_validated = spark.read.parquet("s3://sep152025-trans-project/validated_data/")
# df_validated.printSchema()

# ======================= TRANSFORMATION FUNCTIONS =======================

def extract_month_year(df):
    
    """Extract month and year from transaction_date"""
    
    df = df.withColumn("transaction_month", month(to_date(col("transaction_date"), "yyyy-MM-dd")))
    df = df.withColumn("transaction_year", year(to_date(col("transaction_date"), "yyyy-MM-dd")))
    return df

def categorize_amount(df):
    
    """Categorize amount into bands"""
    
    df = df.withColumn( "amount_range", when(col("transaction_amount") < 500, "Low")
        .when(col("transaction_amount").between(500, 5000), "Medium")
        .otherwise("High")
    )
    return df

def flag_anomaly(df):
    
    """Flag unusually high transaction values"""
    
    df = df.withColumn("is_anomaly", when(col("transaction_amount") > 10000, "Yes").otherwise("No"))
    return df

def derive_risk_score(df):
    
    """Assign risk score based on amount and type"""
    
    df = df.withColumn(
        "risk_score",
        when((col("transaction_amount") >= 10000) & (col("transaction_type") == "debit"), 9)
        .when(col("transaction_amount") >= 5000, 6)
        .otherwise(3)
    )
    return df

def split_txn_id(df):
    
    """Split transaction_id into prefix and number"""
    
    df = df.withColumn("transaction_prefix", split(col("transaction_id"), "-")[0])
    df = df.withColumn("transaction_number", split(col("transaction_id"), "-")[1])
    return df

# def mask_email(df):
#     """Mask email for privacy"""
#     df = df.withColumn("email_masked", regexp_replace(col("email"), "(?<=.{2}).(?=[^@]*?@)", "*"))
#     return df

def add_txn_type_flag(df):
    
    """Add credit/debit flags"""
    
    df = df.withColumn("is_credit", when(col("txn_status") == "credit", lit(1)).otherwise(0))
    df = df.withColumn("is_debit", when(col("txn_status") == "debit", lit(1)).otherwise(0))
    return df
    
# ------------------- Apply transformations -------------------

df_trans = df_validated
# df_trans.show(3)
df_trans = extract_month_year(df_trans)
df_trans = categorize_amount(df_trans)
df_trans = flag_anomaly(df_trans)
df_trans = derive_risk_score(df_trans)
df_trans = split_txn_id(df_trans)
# df_trans = mask_email(df_trans)
df_trans = add_txn_type_flag(df_trans)

# df_trans.write.mode("overwrite").parquet(s3://sep152025-trans-project/df_trans/)

# # ------------------- Add a single "is_invalid" flag -------------------

# # Collect all _check columns automatically

check_columns = [c for c in df_trans.columns if "_check" in c or c == "invalid_foreign_key"]

# # Combine them using OR

condition = " OR ".join([f"{c} = true" for c in check_columns])
df_trans = df_trans.withColumn("is_invalid", expr(condition))

# # ------------------- Separate valid and invalid rows -------------------

valid_df = df_trans.filter("is_invalid = false")
invalid_df = df_trans.filter("is_invalid = true")


# # ------------------- Write to disk -------------------
valid_output = "s3://sep152025-trans-project/transformed_data/valid_transformed/"
invalid_output = "s3://sep152025-trans-project/transformed_data/invalid_transformed/"

valid_df.write.mode("overwrite").parquet(valid_output)
invalid_df.write.mode("overwrite").parquet(invalid_output)



job.commit()