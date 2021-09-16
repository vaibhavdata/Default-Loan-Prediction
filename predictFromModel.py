from sklearn.model_selection import train_test_split
from data_insert.data_loader_predication import Data_Getter
from data_preprocessing.pred_preproessing import Preprocessor
from data_preprocessing import clustering
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger
import pandas as pd
from predication_raw_validation.raw_validation import  Raw_Data_validation



class prediction:

    def __init__(self,path):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Predication_Logs/predicationmodel.txt", 'a+')
        self.pred_data_val = Raw_Data_validation(path)

    def predictionFromModel(self):
        
        # Logging the start of Training
        
        self.log_writer.log(self.file_object, 'Start of predication')
        try:
            self.log_writer.log(self.file_object, "start data getting")
            data_getter =Data_Getter(self.file_object,self.log_writer)
            data =data_getter.get_data()
            self.log_writer.log(self.file_object, "get data succesfullly")
            preprocess =Preprocessor(self.file_object,self.log_writer)
            is_null_present =preprocess.is_null_present(data)
            if is_null_present ==True:
                data = preprocess.impute_missing_values(data)
            
            self.log_writer.log(self.file_object, "import missing value succesfully")
            data = preprocess.row_difference_origin_and_firestpayment(data)
            self.log_writer.log(self.file_object,"inter in label encoding")
            data =preprocess.encoding_source(data)
            data =preprocess.encoding_loan_purpose(data)
            data =preprocess.financial_institution_encoding_country(data)
            self.log_writer.log(self.file_object,"Encoding succesfullly")
            data =preprocess.remove_columns(data,['loan_id','financial_institution','origination_date','first_payment_date','number_of_borrowers'])
            self.log_writer.log(self.file_object,"Remove column succesfullly succesfully")
            data =preprocess.scale_numberical_value(data)
            file_loader = file_methods.File_Operation(self.file_object,self.log_writer)
            #save_model = file_op.save_model(best_model,best_model_name)
            model_name = file_loader.find_correct_model_file()
            model = file_loader.load_model(model_name)
            result = list(model.predict(data))
            predictions =[]
            for res in result:
                if res==0:
                    predictions.append('N')
                else:
                    predictions.append('Y')
            result = pd.DataFrame(predictions,columns=['Predictions'])
            path="Prediction_Output_File/Predictions.csv"
            result.to_csv("Prediction_Output_File/Predictions.csv",header=True) #appends result to prediction file
            self.log_writer.log(self.file_object,'End of Prediction')
            self.file_object.close()
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path