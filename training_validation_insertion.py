from datetime import datetime
from application_logging.logger import App_Logger
from DataTypeValidation_Insertion_Training.DataTypeValidation import DB_Operations
from train_raw_validation.raw_validation import Raw_Data_validation
from DataTransform_Training.datatransform import dataTransform
class train_validation:
    def __init__(self,path):
        self.dbOpration =DB_Operations()
        self.dataTransform =dataTransform()
        self.raw_Data =Raw_Data_validation(path)
        self.file_object =open("Training_Logs/Train_model.txt","a+")
        self.logger =App_Logger()
    def train_validation(self):
        try:
            self.logger.log(self.file_object,"start train validation ")
            LengthOfDateStampInFile,LengthOfTimeStampInFile,column_name,NumberOfColumn =self.raw_Data.value_from_schema()
            regex =self.raw_Data.manaualRegrexCreation()
            self.raw_Data.validationFileName(regex,LengthOfDateStampInFile,LengthOfTimeStampInFile)
            self.raw_Data.validationColumnNumber(NumberOfColumn)
            
            self.logger.log(self.file_object,"Raw Data Validation complite")
            self.logger.log(self.file_object, "Starting Data Transforamtion!!")
            # replacing blanks in the csv file with "Null" values to insert in table
            self.dataTransform.replaceMissingValue()

            self.logger.log(self.file_object, "DataTransformation Completed!!!")
            self.logger.log(self.file_object,"Start Database connection")
            db_object=self.dbOpration.create_db_connection("Loan_Delinauency_Predication")
            collection_object =self.dbOpration.create_collection(db_object)
            self.logger.log(self.file_object,'Creation of Database Completed Successfully !!')
            self.dbOpration.insertion_GoodRawData_into_collection(collection_object)
            self.logger.log(self.file_object,'Insertion of document into collection completed Successfully !!')
            self.raw_Data.deleteExitGoodTrainFile()
            self.logger.log(self.file_object, "Good_Data folder deleted!!!")
            self.logger.log(self.file_object, "Moving bad files to Archive and deleting Bad_Data folder!!!")
            # Move the bad files to archive folder
            self.raw_Data.moveBadFilesToArchiveBad()
            self.logger.log(self.file_object, "Bad files moved to archive!! Bad folder Deleted!!")
            self.logger.log(self.file_object, "Validation Operation completed!!")
            self.logger.log(self.file_object, "Extracting csv file from table")
            
            self.dbOpration.selectDataFromCollection_into_csv(collection_object)
            self.file_object.close()
        except Exception as e:
            raise e