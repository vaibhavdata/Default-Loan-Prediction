import pandas as pd
import numpy as np
from feature_engine.imputation import CategoricalImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import datetime as dt
from collections import Counter
class Preprocessor:
    def __init__(self,file_object,logger_object):
        self.file_object =file_object
        self.logger_object =logger_object

    def is_null_present(self,data):
        self.logger_object.log(self.file_object,"Entered is Null Present in data set")
        self.null_present =False
        try:
            self.null_counts =data.isna().sum()
            for i in self.null_counts:
                if i >0:
                    self.null_present =True
                    break
                if (self.null_present):
                    df_with_null =pd.DataFrame()
                    df_with_null['column'] =data.columns
                    df_with_null['missingValueCount'] =np.array(data.isna().sum())
                    df_with_null.to_csv("preprocessing_data/null_values.csv")
                    self.logger_object.log(self.file_object,"Missing Value Found")
            return self.null_present
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception  Occured while performing is_null_present method %s' % e)
            self.logger_object.log(self.file_object,"Failed to find missing ")
            raise e
    

    def row_difference_origin_and_firestpayment(self,data):
        self.data = data
        self.logger_object.log(self.file_object,'Entered data to Finding Difference time between origination  Time and First payment  Time')

        try:
            self.data['origination_date'] = pd.to_datetime(self.data['origination_date'])
            #self.data['origination_date'] = self.data['origination_date'].dt.strftime('%Y/%m/%d')
            self.data['first_payment_date'] = pd.to_datetime(self.data['first_payment_date'])
            #self.data['origination_date'] = pd.to_datetime(self.data['origination_date'])
            #self.data['first_payment_date'] = pd.to_datetime(self.data['first_payment_date'])
            #self.data['origination_date'] = pd.to_datetime(self.data['origination_date'].dt.strftime('%Y-%m'))
            #self.data['first_payment_date'] = pd.to_datetime(self.data['first_payment_date'].dt.strftime('%Y-%m'))
            #self.data['first_payment_date'] =self.data['first_payment_date'].dt.strftime('%m/%Y')
            orig_date_df = pd.DataFrame()
            first_pay_df = pd.DataFrame()
            tym_difference_df = pd.DataFrame()

            #orig_date_df['o_day'] = self.data['origination_date'].dt.day
            orig_date_df['o_month'] = self.data['origination_date'].dt.month
            #orig_date_df['o_year'] = self.data['origination_date'].dt.year
            #orig_date_df['o_hour'] = self.data['origination_date'].dt.hour
            #orig_date_df['o_min'] = self.data['origination_date'].dt.minute
            #orig_date_df['o_sec'] = self.data['origination_date'].dt.second

            #first_pay_df['f_day'] = self.data['first_payment_date'].dt.day
            first_pay_df['f_month'] = self.data['first_payment_date'].dt.month
            #first_pay_df['f_year'] = self.data['first_payment_date'].dt.year
            #first_pay_df['f_hour'] = self.data['first_payment_date'].dt.hour
            #first_pay_df['f_min'] = self.data['first_payment_date'].dt.minute
            #first_pay_df['f_sec'] = self.data['first_payment_date'].dt.second

            tym_difference_df['d_month'] = first_pay_df['f_month'] - orig_date_df['o_month']
            #tym_difference_df['d_day'] = first_pay_df['f_day'] - orig_date_df['o_day']
            #tym_difference_df['d_year'] = first_pay_df['f_year'] - orig_date_df['o_year']
            #tym_difference_df['d_hour'] = first_pay_df['f_hour'] - orig_date_df['o_hour']
            #tym_difference_df['d_min'] = first_pay_df['f_min'] - orig_date_df['o_min']
            #tym_difference_df['d_sec'] = first_pay_df['f_sec'] - orig_date_df['o_sec']

            
            #tym_difference_df['d_day'] = tym_difference_df['d_day'].apply(lambda x: abs(x))
            tym_difference_df['d_month'] = tym_difference_df['d_month'].apply(lambda x: abs(x))
            #tym_difference_df['d_year'] = tym_difference_df['d_year'].apply(lambda x: abs(x))
            #tym_difference_df['d_hour'] = tym_difference_df['d_hour'].apply(lambda x: abs(x))
            #tym_difference_df['d_min'] = tym_difference_df['d_min'].apply(lambda x: abs(x))
            #tym_difference_df['d_sec'] = tym_difference_df['d_sec'].apply(lambda x: abs(x))
            self.data = pd.concat([self.data, tym_difference_df], axis=1)

            self.logger_object.log(self.file_object,
                                   'Finding Difference between origination and first payment  Time Completed Successfully !!')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Error while finding data differnece origination and first payment Time %s' % str(e))
            raise e

    def financial_institution_encoding_country(self,data):
        self.logger_object.log(self.file_object,'Count financial  institutionStarted dict coding ')
        self.data  = data

        try:
            financial_institution_map = self.data['financial_institution'].value_counts().to_dict()
            self.data['financial_inst_encoding'] = self.data['financial_institution'].map(financial_institution_map)
            self.logger_object.log(self.file_object,'financial institution of financial Feature Successfully Completed')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Error while performing Count Frequency Encoding over Country feature:: %s' % str(e))
            raise e


    def encoding_loan_purpose(self,data):
        self.data = data
        self.logger_object.log(self.file_object,'Entered to Data Row perform One-Hot Encoding on cloud Feature')

        try:
            self.data['loan_purpose'] =self.data['loan_purpose'].map({'A23':1,'B12':2,'C86':3})
                

            self.logger_object.log(self.file_object,'One-Hot Encoding of Source Feature Successfully Completed')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Error while performing Data One-Hot Encoding over Source feature:: %s' % str(e))
            raise e
    def encoding_source(self,data):
        self.data = data
        self.logger_object.log(self.file_object,'Entered to Data Row perform One-Hot Encoding on cloud Feature')

        try:
            self.data['source'] =self.data['source'].map({'X':1,'Y':2,'Z':3})
                

            self.logger_object.log(self.file_object,'One-Hot Encoding of Source Feature Successfully Completed')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Error while performing Data One-Hot Encoding over Source feature:: %s' % str(e))
            raise e


    def separate_label_feature(self,data,label_column_name):
        
        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X=data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
            self.Y=data[label_column_name] # Filter the Label columns
            self.logger_object.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X,self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class')
            raise Exception()
    
    
    def scale_numberical_value(self,data):
        self.logger_object.log(self.file_object,"scale numberical value")
        self.data =data
        self.num_df=self.data[['interest_rate','unpaid_principal_bal', 'loan_term','loan_to_value','debt_to_income_ratio', 'borrower_credit_score','insurance_percent', 'co-borrower_credit_score', 'insurance_type', 'm1',
       'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']]
        try:

            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.num_df)
            self.scaled_num_df = pd.DataFrame(data=self.scaled_data, columns=self.num_df.columns,index=self.data.index)
            self.data.drop(columns=self.scaled_num_df.columns, inplace=True)
            self.data = pd.concat([self.scaled_num_df, self.data], axis=1)

            self.logger_object.log(self.file_object, 'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return self.data

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in scale_numerical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'scaling for numerical columns Failed. Exited the scale_numerical_columns method of the Preprocessor class')
            raise Exception()
    def tranform_value(self,data):
        self.logger_object.log(self.file_object,"tranform data")
        self.data =data
        self.num_df =self.data[['insurance_percent','insurance_type', 'm1',
       'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']]
        try:
            self.scaler = PowerTransformer(method='yeo-johnson')
            self.scaled_data = self.scaler.fit_transform(self.num_df)
            self.scaled_num_df = pd.DataFrame(data=self.scaled_data, columns=self.num_df.columns,index=self.data.index)
            self.data.drop(columns=self.scaled_num_df.columns, inplace=True)
            self.data = pd.concat([self.scaled_num_df, self.data], axis=1)

            self.logger_object.log(self.file_object, 'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in scale_numerical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'scaling for numerical columns Failed. Exited the scale_numerical_columns method of the Preprocessor class')
            raise Exception()



    def remove_columns(self,data,columns):
        self.logger_object.log(self.file_object, 'Entered the remove_columns method of the Preprocessor class')
        self.data=data
        self.columns=columns
        try:
            self.useful_data=self.data.drop(labels=self.columns, axis=1) # drop the labels specified in the columns
            self.logger_object.log(self.file_object,
                                   'Column removal Successful.Exited the remove_columns method of the Preprocessor class')
            return self.useful_data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in remove_columns method of the Preprocessor class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,
                                   'Column removal Unsuccessful. Exited the remove_columns method of the Preprocessor class')
            raise Exception()
    

    
    def remove_unwanted_cols(self,data,return_unwanted_data=False):
        self.logger_object.log(self.file_object,'Removing Unwanted Columns Started !!')
        self.df = data

        try:
            self.data = self.df.drop(['loan_id', 'source', 'financial_institution','origination_date','first_payment_date','loan_purpose'],axis=1)

            if return_unwanted_data == True:
                self.unwanted_data = self.df[['loan_id', 'source', 'financial_institution', 'interest_rate','unpaid_principal_bal', 'loan_term', 'origination_date','first_payment_date', 'loan_to_value', 'number_of_borrowers','debt_to_income_ratio', 'borrower_credit_score', 'loan_purpose','insurance_percent', 'co-borrower_credit_score', 'insurance_type', 'm1','m2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']]

            self.logger_object.log(self.file_object,'Unwanted Columns Deleted Successfully !!')
            self.logger_object.log(self.file_object,'Sample Feature with 1 Row')
            

            if return_unwanted_data == True:
                return self.data,self.unwanted_data
            else:
                return self.data

        except Exception as e:
            self.logger_object.log(self.file_object,'Error occured while removing unwanted columns :: %s' %str(e))
            raise e


    def impute_missing_values(self, data, cols_with_missing_values):
        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        self.data= data
        self.cols_with_missing_values=cols_with_missing_values
        try:
            self.imputer = CategoricalImputer()
            for col in self.cols_with_missing_values:
                self.data[col] = self.imputer.fit_transform(self.data[col])
            self.logger_object.log(self.file_object, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()
    def handle_imbalanced_dataset(self,X,Y):
        self.logger_object.log(self.file_object,
                               'Entered the handle_imbalanced_dataset method of the Preprocessor class')

        try:
            self.rdsmple = RandomOverSampler(sampling_strategy=0.75)
            x_sampled,y_sampled  = self.rdsmple.fit_resample(X,Y)
            self.logger_object.log(self.file_object,'Before Random  Oversampling:: %s' %str(Counter(Y)))
            self.logger_object.log(self.file_object,'After random Oversampling:: %s' %str(Counter(y_sampled)))
            return x_sampled,y_sampled

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in handle_imbalanced_dataset method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'dataset balancing Failed. Exited the handle_imbalanced_dataset method of the Preprocessor class')
            raise Exception()
    def over_sampling_smote(self,x,y):
        self.logger_object.log(self.file_object,'Entered to SMOTE Oversampling Method')
        self.x = x
        self.y = y
        self.smote = SMOTE()
        try:
            x_smote,y_smote = self.smote.fit_resample(x,y)
            self.logger_object.log(self.file_object,'Before SMOTE Oversampling:: %s' %str(Counter(y)))
            self.logger_object.log(self.file_object,'After SMOTE Oversampling:: %s' %str(Counter(y_smote)))
            return x_smote,y_smote
        except Exception as e:
            self.logger_object.log(self.file_object,'Error while performing SMOTE Oversampling %s' %str(e))
            raise e
    def tranform_value_from(self,data):
        self.logger_object.log(self.file_object,"tranform data")
        self.data =data
        self.num_df =self.data[['insurance_percent','insurance_type', 'm1',
       'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11', 'm12']]
        try:
           self.scaled_data =np.cbrt(self.num_df)
           self.scaled_num_df = pd.DataFrame(data=self.scaled_data, columns=self.num_df.columns,index=self.data.index)
           self.data.drop(columns=self.scaled_num_df.columns, inplace=True)
           self.data = pd.concat([self.scaled_num_df, self.data], axis=1)
           self.logger_object.log(self.file_object, 'transform for numerical values successful. Exited the transform_numerical_columns method of the Preprocessor class')
           return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in scale_numerical_columns method of the Preprocessor class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object, 'scaling for numerical columns Failed. Exited the scale_numerical_columns method of the Preprocessor class')
            raise Exception()
