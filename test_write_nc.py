from pyspark.sql import SparkSession
import xarray as xr
import os
import pyarrow.fs as fs
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("master") \
        .config("spark.hadoop.hadoop.security.authentication", "kerberos") \
        .config("spark.hadoop.hadoop.security.authorization", "true") \
        .config("spark.security.credentials.hive.enabled","false") \
        .config("spark.security.credentials.hbase.enabled","false") \
        .enableHiveSupport().getOrCreate()
    
    # Path file di HDFS
    hdfs_path = 'hdfs://master-01.bnpb.go.id:8020/user/warehouse/JAXA/curah_hujan/2021/12/06/gsmap_now_rain.20211206.0000.nc'
    local_path = 'data/gsmap/gsmap_now_rain.20211206.0000.nc'

    # Mengakses FileSystem melalui JVM
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)

    # Membuat objek Path di HDFS dan lokal
    hdfs_file_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_path)
    local_file_path = spark._jvm.org.apache.hadoop.fs.Path(local_path)

    # Gunakan FileUtil untuk menyalin dari HDFS ke sistem lokal
    spark._jvm.org.apache.hadoop.fs.FileUtil.copy(fs, hdfs_file_path, spark._jvm.org.apache.hadoop.fs.FileSystem.getLocal(hadoop_conf), local_file_path, False, hadoop_conf)

    print(f"File berhasil disalin ke: {local_path}")