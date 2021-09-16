import pickle
import os
import shutil


class File_Operation:
    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.model_directory='models/'

    def save_model(self,model,filename):
        self.logger_object.log(self.file_object, 'Entered the save_model method of the File_Operation class')
        try:
            path = os.path.join(self.model_directory,filename) #create seperate directory for each cluster
            if os.path.isdir(path): #remove previously existing models for each clusters
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path) #
            with open(path +'/' + filename+'.sav',
                      'wb') as f:
                pickle.dump(model, f) # save the model to file
            self.logger_object.log(self.file_object,
                                   'Model File '+filename+' saved. Exited the save_model method of the Model_Finder class')

            return 'success'
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in save_model method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Model File '+filename+' could not be saved. Exited the save_model method of the Model_Finder class')
            raise Exception()

    def load_model(self,filename):
        self.logger_object.log(self.file_object, 'Entered the load_model method of the File_Operation class')
        try:
            with open(self.model_directory + filename + '/' + filename + '.sav',
                      'rb') as f:
                self.logger_object.log(self.file_object,
                                       'Model File ' + filename + ' loaded. Exited the load_model method of the Model_Finder class')
                return pickle.load(f)
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in load_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model File ' + filename + ' could not be saved. Exited the load_model method of the Model_Finder class')
            raise Exception()

    def find_correct_model_file(self):
        self.logger_object.log(self.file_object,'Entered the find_correct_model_file method of the File_Operation class')
        try:
            
            self.folder_name = self.model_directory
            model_dir = os.listdir(self.model_directory)
            model_name = os.listdir(self.model_directory + model_dir[0])
            model_name = model_name[0].split('.')[0]
            self.logger_object.log(self.file_object,'Exited the find_correct_model_file method of FileMethods Package')
            return model_name
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception Occured in finding_correct_model_file method :: %s' % str(e))
            self.logger_object.log(self.file_object,'Exited the find_correct_model_file method of FileMethods Package')
            raise e