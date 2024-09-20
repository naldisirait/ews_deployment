import re
from datetime import datetime, timedelta
import os

os.environ['HADOOP_HOME'] = '/etc/hadoop'
os.environ['HADOOP_CONF_DIR'] = '/etc/hadoop/conf'

# Function to extract the timestamp using regex
def extract_timestamp(file_path):
    # Regular expression to extract the timestamp from the file path
    timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})')
    match = timestamp_pattern.search(file_path)
    if match:
        date_str = match.group(1)  # YYYY-MM-DD
        time_str = match.group(2)  # HH-MM-SS
        timestamp_str = f'{date_str} {time_str}'
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H-%M-%S')  # Convert to datetime object
        return timestamp
    else:
        return None
        
import re
from datetime import datetime, timedelta

# Function to extract the timestamp using regex
def extract_timestamp(file_path):
    # Regular expression to extract the timestamp from the file path
    timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})')
    match = timestamp_pattern.search(file_path)
    if match:
        date_str = match.group(1)  # YYYY-MM-DD
        time_str = match.group(2)  # HH-MM-SS
        timestamp_str = f'{date_str} {time_str}'
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H-%M-%S')  # Convert to datetime object
        return timestamp
    else:
        return None
        
def check_last_hours_data(file_paths, hours):
    """
    Function to checks if data is available for the last `N hours`. If all data is available, it returns the list of file paths for 
    those hours.If any data is missing, it reports the missing hours.
    
    Args:
    file_paths (list): List of file paths
    hours (int): Integer defining the number of hours to check 
    
    Returns: 
        tuple (str, list): List of file paths for the last `hours` if all data is available, or missing hours if not.
    """
    # Get current time and the time 'hours' ago
    current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = current_time - timedelta(hours=hours)
    print(f"current_time : {current_time }")
    print(f"start_time : {start_time}")

    # Collect all timestamps and map them to their corresponding file paths
    available_files = [(extract_timestamp(path), path) for path in file_paths if extract_timestamp(path) is not None]

    # Filter only the files within the last 'hours'
    files_in_last_hours = [
        file_path for (timestamp, file_path) in available_files if start_time <= timestamp <= current_time
    ]
    files_in_last_hours = files_in_last_hours[-hours:]
    # Check if all the expected hours are available
    missing_hours = []
    for hour in range(hours + 1):  # Include the current hour
        expected_time = start_time + timedelta(hours=hour)
        if expected_time not in [ts for ts, path in available_files]:
            missing_hours.append(expected_time)

    # Return results
    if missing_hours:
        print("Data is missing for the following hours:")
        for missing_time in missing_hours:
            print(missing_time.strftime('%Y-%m-%d %H:%M:%S'))
        return ("Not available", missing_hours) # Indicate that not all data is available
    else:
        #print(f"All data for the last {hours} hours is available.")
        return ("Available", files_in_last_hours)
    
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

def get_precipitation_from_big_lake(hours):
    """
    Function to get precipitation data from big lake, it will return the last hours if the data is available, 
    Args:
        hours (int): Number of hours to look back
    Returns:

    
    """

    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("master") \
        .config("spark.hadoop.hadoop.security.authentication", "kerberos") \
        .config("spark.hadoop.hadoop.security.authorization", "true") \
        .config("spark.security.credentials.hive.enabled","false") \
        .config("spark.security.credentials.hbase.enabled","false") \
        .enableHiveSupport().getOrCreate()
    
    # Define the HDFS path
    hdfs_path = "/user/warehouse/SPLP/PUPR/curah_hujan/palu"
    # hdfs_path = "/user/warehouse/SPLP/PUPR"
    # hdfs_path = "/user/warehouse/JAXA/curah_hujan"
    # List HDFS files recursively
    hdfs_files = list_hdfs_files_recursive(spark, hdfs_path)

    json_files = [i for i in hdfs_files if (".json" in i and "curah_hujan" in i) and ("unstructed" not in i)]
   
    # Check if data is available for the last N hours
    result = check_last_hours_data(json_files, 10)
