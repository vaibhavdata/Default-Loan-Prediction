
import shutil
import pymongo
import os
import pandas as pd
from application_logging.logger import App_Logger

class DB_Operations:
    def __init__(self):
        self.client = pymongo.MongoClient('mongodb://127.0.0.1:27017')
        self.path = 'Training_Database'
        self.good_file_path = 'Prediction_Raw_files_validated/Good_Raw'
        self.bad_file_path = 'Prediction_Raw_files_validated/Bad_Raw'
        self.FileFromDB = 'Prediction_FileFromDB'
        self.logger = App_Logger()

    def create_db_connection(self,database_name):
        file = open('Predication_Logs/General_Log.txt', 'a+')
        self.logger.log(file,'Entered create_db_connection() method of DB_Operation class of training_db_operations package')
        file.close()

        try:
            self.db_object = self.client[str(database_name)]
            file = open("Predication_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Database < %s > Created successfully!!" % str(database_name))
            self.logger.log(file, "Database < %s > Connected successfully!!" % str(database_name))
            file.close()

            file = open('Predication_Logs/DataBaseConnectionLog.txt', 'a+')
            self.logger.log(file,'Successfully Executed create_db_connection() method of DB_Operation class of db_operation package')
            file.close()
            return self.db_object
        except Exception as ex:
            file = open("Predication_Logs/General_Log.txt", 'a+')
            self.logger.log(file, 'Error while settingup connection with database.Error :: %s' % ex)
            file.close()


    def create_collection(self,db_object):
        file = open('Predication_Logs/General_Log.txt', 'a+')
        self.logger.log(file,'Entered create_collection() method of DB_Operation class of training_db_operations package')
        file.close()

        try:
            collection_list = db_object.collection_names()
            collection_name = 'GoodRawData'
            file = open("Predication_Logs/CreateCollectionLog.txt", 'a+')
            if collection_name in collection_list:
                collection_object = db_object[collection_name]
                collection_object.remove({})
                self.logger.log(file,'GoodRawData Collection already Exist and deleted documents Successfully!!')
            else:
                collection_object = db_object.create_collection(collection_name)
                self.logger.log(file,'GoodRawData Collection created Successfully !!')

            file.close()
            file = open('Predication_Logs/General_Log.txt', 'a+')
            self.logger.log(file,'Successfully Executed create_collection() method of DB_Operation class of db_operation package')
            file.close()
            return collection_object
        except Exception as ex:
            file = open("Predication_Logs/General_Log.txt", 'a+')
            self.logger.log(file, 'Error while creating collection in database.Error :: %s' % ex)
            file.close()

    
    def insertion_GoodRawData_into_collection(self,collection_object):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger.log(file, 'Entered insertionGoodData_into_collection() method of DB_Operation class of training_db_operations package')
        file.close()

        try:
            only_files = [file for file in os.listdir(self.good_file_path)]
            file = open("Training_Logs/DataBaseSelectionLog.txt", 'a+')
            for f in only_files:
                
                data = pd.read_csv(os.path.join(self.good_file_path,f))
                document = [{'loan_id':id,'source':source,'financial_institution':financial,'interest_rate':interest,'unpaid_principal_bal':unpaid,'loan_term':loan_term,'origination_date':origination_date,'first_payment_date':first_pay,'loan_to_value':loan_to_val,'number_of_borrowers':number_of,'debt_to_income_ratio':debt_to,'borrower_credit_score':borrower,'loan_purpose':loan_purp,'insurance_percent':insurance_pre,'co-borrower_credit_score':co_borr,'insurance_type':insurance,'m1':m1,'m2':m2,'m3':m3,'m4':m4,'m5':m5,'m6':m6,'m7':m7,'m8':m8,'m9':m9,'m10':m10,'m11':m11,'m12':m12} for id,source,financial,interest,unpaid,loan_term,origination_date,first_pay,loan_to_val,number_of,debt_to,borrower,loan_purp,insurance_pre,co_borr,insurance,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12 in zip(data['loan_id'],data['source'],data['financial_institution'],data['interest_rate'],data['unpaid_principal_bal'],data['loan_term'],data['origination_date'],data['first_payment_date'],data['loan_to_value'],data['number_of_borrowers'],data['debt_to_income_ratio'],data['borrower_credit_score'],data['loan_purpose'],data['insurance_percent'],data['co-borrower_credit_score'],data['insurance_type'],data['m1'],data['m2'],data['m3'],data['m4'],data['m5'],data['m6'],data['m7'],data['m8'],data['m9'],data['m10'],data['m11'],data['m12'])]
                collection_object.insert_many(document)
                
                self.logger.log(file,'Data File :: %s Inserted Successfully in Collection'.format(f))

            file.close()
            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger.log(file,'Successfully Executed insertion_GoodRawData_into_collection() method of DB_Operation class of db_operation package')
            file.close()
        except Exception as ex:
            file = open("Training_Logs/General_Log.txt", 'a+')
            self.logger.log(file, 'Error while inserting data file into collection.Error :: %s' % ex)
            file.close()
    

    def selectDataFromCollection_into_csv(self, collection_object):
        file = open('Training_Logs/General_Log.txt', 'a+')
        self.logger.log(file,'Entered selectDataFromCollection_into_csv() method of DB_Operation class of training_db_operations package')
        file.close()

        try:
            data = list()
            for i in collection_object.find():
                data.append({'loan_id':i['loan_id'],'source':i['source'],'financial_institution':i['financial_institution'],'interest_rate':i['interest_rate'],'unpaid_principal_bal':i['unpaid_principal_bal'],'loan_term':i['loan_term'],'origination_date':i['origination_date'],'first_payment_date':i['first_payment_date'],'loan_to_value':i['loan_to_value'],'number_of_borrowers':i['number_of_borrowers'],'debt_to_income_ratio':i['debt_to_income_ratio'],'borrower_credit_score':i['borrower_credit_score'],'loan_purpose':i['loan_purpose'],'insurance_percent':i['insurance_percent'],'co-borrower_credit_score':i['co-borrower_credit_score'],'insurance_type':i['insurance_type'],'m1':i['m1'],'m2':i['m2'],'m3':i['m3'],'m4':i['m4'],'m5':i['m5'],'m6':i['m6'],'m7':i['m7'],'m8':i['m8'],'m9':i['m9'],'m10':i['m10'],'m11':i['m11'],'m12':i['m12']})
    
    

            if not os.path.isdir(self.FileFromDB):
                os.makedirs(self.FileFromDB)

            dataframe = pd.DataFrame(data,columns=['loan_id', 'source', 'financial_institution','interest_rate', 'unpaid_principal_bal','loan_term', 'origination_date', 'first_payment_date','loan_to_value', 'number_of_borrowers','debt_to_income_ratio', 'borrower_credit_score','loan_purpose', 'insurance_percent', 'co-borrower_credit_score','insurance_type', 'm1', 'm2', 'm3','m4', 'm5','m6','m7','m8','m9','m10','m11','m12'])
            dataframe.to_csv(os.path.join(self.FileFromDB,'InputFile.csv'),index=False)

            file = open("Training_Logs/DataBase_Into_CSVLog.txt", 'a+')
            self.logger.log(file,'CSV File Exported Successfully !!!')
            file.close()

            file = open('Training_Logs/General_Log.txt', 'a+')
            self.logger.log(file,'Successfully Executed selectDataFromCollection_into_csv() method of DB_Operation class of db_operation package')
            file.close()
        except Exception as ex:
            file = open("Training_Logs/General_Log.txt", 'a+')
            self.logger.log(file, 'Error while selecting data file and store it as csv.Error :: %s' % ex)
            file.close()







