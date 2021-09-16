from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from data_insert.data_loader import Data_Getter
from data_preprocessing.preprocessing import Preprocessor
from data_preprocessing import clustering
from best_model_finder import tuner
from best_model_finder import tunerss
from file_operations import file_methods
from application_logging import logger
#Creating the common Logging object


class TrainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

    def trainingModel(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # Getting the data from the source
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
            self.log_writer.log(self.file_object,"label encoding")
            data =preprocess.encoding_source(data)
            data =preprocess.encoding_loan_purpose(data)
            data =preprocess.financial_institution_encoding_country(data)
            self.log_writer.log(self.file_object,"Encoding succesfullly")
            data =preprocess.remove_columns(data,['loan_id','financial_institution','origination_date','first_payment_date','number_of_borrowers'])
            #data =preprocess.tranform_value(data)
            data =preprocess.tranform_value_from(data)
            
            self.log_writer.log(self.file_object,"Remove column succesfullly succesfully")
            self.log_writer.log(self.file_object,"separate label  feature starting")
            X,Y=preprocess.separate_label_feature(data,label_column_name='m13')
        

            # splitting the data into training and test set for each cluster one by one
            x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=1 / 3, random_state=355)
            x_train = preprocess.scale_numberical_value(x_train)
            x_test = preprocess.scale_numberical_value(x_test)
            x_train,y_train =preprocess.handle_imbalanced_dataset(x_train,y_train)
            #x_train,y_train =preprocess.over_sampling_smote(x_train,y_train)
            #self.log_writer.log(self.file_object,print(x_train.skew()))
                
            model_finder=tunerss.Model_Finder(self.file_object,self.log_writer) # object initialization

            #getting the best model for each of the clusters
            best_model_name,best_model=model_finder.get_best_model(x_train,y_train,x_test,y_test)

            #saving the best model to the directory.
            file_op = file_methods.File_Operation(self.file_object,self.log_writer)
            save_model=file_op.save_model(best_model,best_model_name)

            self.log_writer.log(self.file_object,'Successfull End of Training')
            self.file_object.close()
            

        except Exception:
            self.log_writer.log(self.file_object,"Error in training")
            raise Exception