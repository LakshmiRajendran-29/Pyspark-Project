import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, to_date, length, regexp_extract, count, when, lit, month, year,split,trim,regexp_replace,expr
from pyspark.sql.window import Window

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

dynamic_frame = glueContext.create_dynamic_frame.from_catalog(
    database="sep15_database",
    table_name="projectincoming"
)

df = dynamic_frame.toDF()

raw_df = df.withColumn("transaction_date", to_date(col("transaction_date"), "dd/MM/yyyy")) \
               .withColumn("txn_date", to_date(col("txn_date"), "dd/MM/yyyy"))
               
validated_df = raw_df

def null_checks_cols(df,columns):
    for column_name in columns:
        df = df.withColumn(f"{column_name}_null_check",col(column_name).isNull())
    return df

mandatory_Columns = ["transaction_id","account_number","transaction_amount","transaction_date","transaction_type","currency"]
str_column = ["transaction_id","account_number"]

def check_mandatory_exists(df,required_columns):
    missing = [column_name for column_name in required_columns if column_name not in df.columns ]
    if missing:
        raise ValueError(f"Missing Mandatory Column{missing}")
    return df

def check_empty_string(df, columns):
    for column_name in columns:
        df=df.withColumn(f"{column_name}_check_empty_string",(col(column_name).isNull()) | (col(column_name) == ""))
    return df

def check_numeric_column_range(df,column,min_val, max_val):
    df = df.withColumn(f"range_check_{column}",((col(column).isNull()) | (col(column) < min_val) | (col(column) > max_val)))
    return df

def check_string_length(df,columns,max_len):
    for column in columns:
        df = df.withColumn(f"{column}_str_length",length(col(column)) > max_len )
    return df

def check_regex_pattern(df,column,pattern):
    df = df.withColumn(f"{column}_check_regex",regexp_extract(col(column),pattern,0) == "")
    return  df

# 	•	If regex does not match → returns "". Condition = True (invalid).
# 	•	If regex matches → returns full string. Condition = False (valid).


def check_invalid_currency_codes(df,column):
    valid_currency = ["USD","INR","GBP","EUR"]
    df = df.withColumn(f"{column}_check_invalid_currency", ~col(column).isin(valid_currency))
    return df

def check_invalid_transaction_type(df,column):
    valid_types = ["debit","credit","transfer"]
    df = df.withColumn(f"{column}_check_invalid_transaction_type", ~col(column).isin(valid_types))
    return df


def check_duplicate_rows(df,column):
    df=df.withColumn("is_duplicate",count("*").over(Window.partitionBy(column))> 1)
    return df


def check_amount_sign(df,amount_col,trans_type):
    df = df.withColumn(f"{trans_type}_check_amount_sign",
                       when( (col(trans_type) == "debit") & (col(amount_col) > 0),lit(True))
                       .when( (col(trans_type) == "crdit") & (col(amount_col) < 0),lit(True))
                       .otherwise(lit(False)))
    return df

def check_leading_trailing_spaces(df, columns):
    for c in columns:
        df = df.withColumn(f"{c}_space_check", col(c) != trim(col(c)))
    return df

def check_phone_format(df, column):
    return df.withColumn(f"{column}_invalid_phone", (~col(column).rlike(r'^[0-9]{10}$')) | (col(column).isNull() ))

def check_foreign_key(df,customer_df,trans_cus_id,customer_cus_id):
    df = df.join(customer_df.select(customer_cus_id).distinct(), on = df[trans_cus_id]  == customer_df[customer_cus_id], how = "left_anti" ) \
    .withColumn("invalid_foreign_key", lit(True))
    return df


def check_email_pattern(df,column):
    email_pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-za-z0-9]{2,}$"
    df = df.withColumn(f"{column}_invalid",regexp_extract(col(column),email_pattern,0) == "")
    return df

# validated_df = null_check(validated_df,col("transaction_type"))
validated_df = null_checks_cols(validated_df,mandatory_Columns)

# check_mandatory_exists function will not return anything
validated_df = check_mandatory_exists(validated_df,mandatory_Columns)
validated_df = check_empty_string(validated_df,["transaction_id","account_number","transaction_type","currency"])
validated_df = check_numeric_column_range(validated_df,"transaction_amount",1,100000)
validated_df = check_string_length(validated_df,str_column,7)
validated_df = check_regex_pattern(validated_df,"transaction_id",r'^[A-Z0-9]{6}$')
validated_df = check_regex_pattern(validated_df,"account_number",r'^[A-Z0-9]{7}$')
validated_df = check_invalid_currency_codes(validated_df,"currency")
validated_df = check_duplicate_rows(validated_df,"transaction_id")
# validated_df = check_foreign_key(validated_df,customer_df,"customer_id","customer_id")

validated_df = check_leading_trailing_spaces(validated_df, mandatory_Columns)
# validated_df.show(30)

# customer_df = check_phone_format(customer_df,"phone")
# customer_df = check_email_pattern(customer_df, "email")
# customer_df.show()
# Combine all _check columns into a single condition
check_columns = [c for c in validated_df.columns if "_check" in c ]     # returns Column name with check
condition = " OR ".join([f"{c} = true" for c in check_columns])      # checks the TRUE value in rows of check_columns
invalid_df = validated_df.filter(condition)   #  filter the all the rows which have the true called as Invalid data


invalid_df = invalid_df.write \
    .option("header", True) \
    .mode("overwrite") \
  .parquet("s3://sep152025-trans-project/invalid_data/")

# Filter valid rows
validated_df = validated_df.filter(" OR ".join([f"{c} = false" for c in validated_df.columns if "_check" in c]))

validated_df.write \
        .option("header",True) \
        .mode("overwrite") \
        .parquet("s3://sep152025-trans-project/validated_data/")


job.commit()