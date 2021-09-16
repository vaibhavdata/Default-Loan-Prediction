from datetime import datetime
from os import listdir
import pandas as pd
from application_logging.logger import App_Logger

class dataTransform:
    def __init__(self):
        self.goodDataPath ="Prediction_Raw_files_validated/Good_Raw"
        self.logger =App_Logger()

    def replaceMissingValue(self):
        log_file = open("Predication_Logs/dataTransformLog.txt", 'a+')
        try:
            onlyfiles = [f for f in listdir(self.goodDataPath)]
            for file in onlyfiles:
                csv = pd.read_csv(self.goodDataPath+"/" + file)
                csv.fillna('NULL',inplace=True)
                    
                    
                csv.to_csv(self.goodDataPath+ "/" + file, index=None, header=True)
                self.logger.log(log_file," %s: File Transformed successfully!!" % file)
               #log_file.write("Current Date :: %s" %date +"\t" + "Current time:: %s" % current_time + "\t \t" +  + "\n")
        except Exception as e:
            self.logger.log(log_file, "Data Transformation failed because:: %s" % e)
               #log_file.write("Current Date :: %s" %date +"\t" +"Current time:: %s" % current_time + "\t \t" + "Data Transformation failed because:: %s" % e + "\n")
            log_file.close()
        log_file.close()