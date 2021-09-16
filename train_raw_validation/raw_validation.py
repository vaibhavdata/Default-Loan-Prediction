from os import listdir
import pandas as pd
import csv
from datetime import datetime
import shutil
import json
import os
import re
from application_logging.logger import App_Logger

class Raw_Data_validation:
    def __init__(self,path):
        self.Batch_dir =path
        self.schema_path ="schema_Training.json"
        self.logger =App_Logger()
    def value_from_schema(self):
        try:
            with open(self.schema_path,'r') as f:
                dic =json.load(f)
                f.close()
            patteren = dic['SampleFileName']
            LengthOfDateStampInFile =dic['LengthOfDateStampInFile']
            LengthOfTimeStampInFile =dic['LengthOfTimeStampInFile']
            column_name =dic['ColName']
            NumberOfColumn = dic['NumberofColumns']

            file =open("Training_Logs/valueFromSchema.txt","a+")
            message = "LengthOfDateStampInFile:: %s" % LengthOfDateStampInFile+ "\t" + "LengthOfTimeStampInFile:: %s" % LengthOfTimeStampInFile + "\t " + "NumberofColumns:: %s" % NumberOfColumn+ "\n"
            self.logger.log(file,message)
            file.close()
        except ValueError:
            file =open("Training_Logs/valueFromSchema.txt","a+")
            self.logger.log(file,"value Error in train value from schema json file")
            file.close()
            raise ValueError()
        except KeyError:
            file =open("Training_Logs/valueFromSchema.txt","a+")
            self.logger.log(file,"key  Error in train value from  schema json file")
            
        except Exception as e:
            file =open("Training_Logs/valueFromSchema.txt","a+")
            self.logger.log(file,str(e))
            file.close()
        
            raise e
        return LengthOfDateStampInFile,LengthOfTimeStampInFile,column_name,NumberOfColumn
    def manaualRegrexCreation(self):
        regex = "['LoanDelinquency']+['\_'']+[\d_]+[\d]+\.csv"
        return regex
    def createDiretorGoodBadROwData(self):
        try:
            path = os.path.join("Training_Raw_files_validated/", "Good_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)
            path = os.path.join("Training_Raw_files_validated/", "Bad_Raw/")
            if not os.path.isdir(path):
                os.makedirs(path)

        except OSError as ex:
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file,"Error while creating Directory %s:" % ex)
            file.close()
            raise OSError
    
    def deleteExitGoodTrainFile(self):
        try:
            path = 'Training_Raw_files_validated/'
            if os.path.isdir(path + 'Good_Raw/'):
                shutil.rmtree(path + 'Good_Raw/')
                file = open("Training_Logs/GeneralLog.txt", 'a+')
                self.logger.log(file,"GoodRaw directory deleted successfully!!!")
                file.close()
        except OSError as s:
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file,"Error while Deleting Directory : %s" %s)
            file.close()
            raise OSError
    def deleteExitBadTrainFile(self):
        try:
            path = 'Training_Raw_files_validated/'
            if os.path.isdir(path + 'Bad_Raw/'):
                shutil.rmtree(path + 'Bad_Raw/')
                file = open("Training_Logs/GeneralLog.txt", 'a+')
                self.logger.log(file,"BadRaw directory deleted before starting validation!!!")
                file.close()
        except OSError as s:
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file,"Error while Deleting Directory : %s" %s)
            file.close()
            raise OSError
    
    def validationFileName(self,regex,LengthOfDateStampInFile,LengthOfTimeStampInFile):
        self.deleteExitBadTrainFile()
        self.deleteExitGoodTrainFile()
        self.createDiretorGoodBadROwData()
        onlyfiles =[f for f in listdir(self.Batch_dir)]
        try:
            f = open("Training_Logs/nameValidationLog.txt", 'a+')
            for filename in onlyfiles:
                if (re.match(regex, filename)):
                    splitAtDot = re.split('.csv', filename)
                    splitAtDot = (re.split('_', splitAtDot[0]))
                    if len(splitAtDot[1]) == LengthOfDateStampInFile:
                        if len(splitAtDot[2]) == LengthOfTimeStampInFile:
                            shutil.copy("train_Batch_File/" + filename, "Training_Raw_files_validated/Good_Raw")
                            self.logger.log(f,"Valid File name!! File moved to GoodRaw Folder :: %s" % filename)
                        else:
                            shutil.copy("train_Batch_File/" + filename, "Training_Raw_files_validated/Bad_Raw")
                            self.logger.log(f,"Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
                    else:
                        shutil.copy("train_Batch_File/" + filename, "Training_Raw_files_validated/Bad_Raw")
                        self.logger.log(f,"Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)
                else:
                    shutil.copy("train_Batch_File/" + filename, "Training_Raw_files_validated/Bad_Raw")
                    self.logger.log(f, "Invalid File Name!! File moved to Bad Raw Folder :: %s" % filename)

            f.close()

        except Exception as e:
            f = open("Training_Logs/nameValidationLog.txt", 'a+')
            self.logger.log(f, "Error occured while validating FileName %s" % e)
            f.close()
            raise e
    def validationColumnNumber(self,NumberOfColumn):
        try:
            f = open("Training_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f,"Column Length Validation Started!!")
            for file in listdir('Training_Raw_files_validated/Good_Raw/'):
                csv = pd.read_csv("Training_Raw_files_validated/Good_Raw/" + file)
                if csv.shape[1] == NumberOfColumn:
                    pass
                else:
                    shutil.move("Training_Raw_files_validated/Good_Raw/" + file, "Training_Raw_files_validated/Bad_Raw")
                    self.logger.log(f, "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
            self.logger.log(f, "Column Length Validation Completed!!")
        except OSError:
            f = open("Training_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f, "Error Occured while moving the file :: %s" % OSError)
            f.close()
            raise OSError
        except Exception as e:
            f = open("Training_Logs/columnValidationLog.txt", 'a+')
            self.logger.log(f, "Error Occured:: %s" % e)
            f.close()
            raise e
        f.close()
    def moveBadFilesToArchiveBad(self):
        
        now = datetime.now()
        date = now.date()
        time = now.strftime("%H%M%S")
        try:

            source = 'Training_Raw_files_validated/Bad_Raw/'
            if os.path.isdir(source):
                path = "TrainingArchiveBadData"
                if not os.path.isdir(path):
                    os.makedirs(path)
                dest = 'TrainingArchiveBadData/BadData_'+str(date)+"_"+str(time)
                if not os.path.isdir(dest):
                    os.makedirs(dest)
                files = os.listdir(source)
                for f in files:
                    if f not in os.listdir(dest):
                        shutil.move(source + f, dest)
                file = open("Training_Logs/GeneralLog.txt", 'a+')
                self.logger.log(file,"Bad files moved to archive")
                path = 'Training_Raw_files_validated/'
                if os.path.isdir(path + 'Bad_Raw/'):
                    shutil.rmtree(path + 'Bad_Raw/')
                self.logger.log(file,"Bad Raw Data Folder Deleted successfully!!")
                file.close()
        except Exception as e:
            file = open("Training_Logs/GeneralLog.txt", 'a+')
            self.logger.log(file, "Error while moving bad files to archive:: %s" % e)
            file.close()
            raise e



        



                


        
    


        




