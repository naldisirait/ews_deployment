from pyspark.sql import SparkSession
import xarray as xr
import os
import pyarrow.fs as fs
from pyspark.sql import SparkSession

def list_hdfs_files_recursive(spark, path):
    hadoop = spark._jvm.org.apache.hadoop
    fs = hadoop.fs.FileSystem
    conf = hadoop.conf.Configuration()
    conf.set("fs.defaultFS", "hdfs://master-01.bnpb.go.id:8020")
    files = []
    
    def recursive_list_files(path):
        try:
            for f in fs.get(conf).listStatus(path):
                files.append(str(f.getPath()))
                if fs.get(conf).isDirectory(f.getPath()):
                    recursive_list_files(f.getPath())
        except Exception as e:
            print("Error:", e)
    
    recursive_list_files(hadoop.fs.Path(path))
    
    return files

def run():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("master") \
        .config("spark.hadoop.hadoop.security.authentication", "kerberos") \
        .config("spark.hadoop.hadoop.security.authorization", "true") \
        .config("spark.security.credentials.hive.enabled","false") \
        .config("spark.security.credentials.hbase.enabled","false") \
        .enableHiveSupport().getOrCreate()
        
    # Define the HDFS path
    # hdfs_path = "/user/warehouse/SPLP/PUPR/curah_hujan/palu"
    # hdfs_path = "/user/warehouse/SPLP/PUPR"
    hdfs_path = "/user/warehouse/JAXA/curah_hujan"

    # List HDFS files recursively
    hdfs_files = list_hdfs_files_recursive(spark, hdfs_path)

    nc_files = [i for i in hdfs_files if (".nc" in i)]

    # Path file di HDFS
    # hdfs_path = 'hdfs://master-01.bnpb.go.id:8020/user/warehouse/JAXA/curah_hujan/2021/12/06/gsmap_now_rain.20211206.0000.nc'
    hdfs_path = nc_files[2]

    local_path = 'data/gsmap/gsmap_now_rain.nc'

    # Mengakses FileSystem melalui JVM
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)

    # Membuat objek Path di HDFS dan lokal
    hdfs_file_path = spark._jvm.org.apache.hadoop.fs.Path(hdfs_path)
    local_file_path = spark._jvm.org.apache.hadoop.fs.Path(local_path)

    # Gunakan FileUtil untuk menyalin dari HDFS ke sistem lokal
    spark._jvm.org.apache.hadoop.fs.FileUtil.copy(fs, hdfs_file_path, spark._jvm.org.apache.hadoop.fs.FileSystem.getLocal(hadoop_conf), local_file_path, False, hadoop_conf)

    print(f"File berhasil disalin ke: {local_path}")