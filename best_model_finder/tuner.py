from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score,f1_score,recall_score,precision_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, make_scorer

class Model_Finder:
    

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.clf = RandomForestClassifier()
        self.xgb = XGBClassifier(use_label_encoder=False,objective='binary:logitraw',eval_metric = "logloss")
        self.svm =SVC()
        self.dTree =DecisionTreeClassifier()

    def get_best_params_for_svm(self,train_x,train_y):
        
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_svm method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"kernel": ['rbf'],
                          "C": [0.2],
                          "random_state": [100]}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.svm, param_grid=self.param_grid, cv=2,  verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.kernel = self.grid.best_params_['kernel']
            self.C = self.grid.best_params_['C']
            self.random_state = self.grid.best_params_['random_state']


            #creating a new model with the best parameters
            self.svm = SVC(kernel=self.kernel,C=self.C,random_state=self.random_state)
            # training the mew model
            self.svm.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'SVM best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_svm method of the Model_Finder class')

            return self.svm
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_svm method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'SVM training  failed. Exited the get_best_params_for_svm method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):

        
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                'learning_rate': [0.02],
                'max_depth': [10],
                'n_estimators': [1000]

            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(estimator=self.xgb, param_grid=self.param_grid_xgboost, verbose=3,cv=2,scoring='f1_micro')
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_dessionTree(self,train_x,train_y):
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_DecisionTree method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_Dtree = {

                'min_samples_split': [2, 3],
                'max_leaf_nodes': [5,10],
                "random_state": [50],
                "max_features": ['sqrt']


            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(estimator=self.dTree, param_grid=self.param_grid_Dtree, verbose=1,cv=2,scoring='f1_micro')
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.min_samples_split = self.grid.best_params_['min_samples_split']
            self.max_leaf_nodes = self.grid.best_params_['max_leaf_nodes']
            self.random_state = self.grid.best_params_['random_state']
            self.max_features =self.grid.best_params_['max_features']

            # creating a new model with the best parameters
            self.dTree = DecisionTreeClassifier(min_samples_split=self.min_samples_split, max_leaf_nodes=self.max_leaf_nodes, random_state=self.random_state,max_features=self.max_features)
            # training the mew model
            self.dTree.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Decision Tree best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_Decision Tree method of the Model_Finder class')
            return self.dTree
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_Desision Tree method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for Decision Tree method of the Model_Finder class')
            raise Exception()

    

    def get_best_model(self,train_x,train_y,test_x,test_y):

        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for XGBoost
        try:
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x) # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, 'ROC for XGBoost:' + str(self.xgboost_score))  # Log AUC
            else:
                self.xgboost_f1_score = f1_score(test_y, self.prediction_xgboost,average='micro') # AUC for XGBoost
                self.xgb_confiux_matrix =confusion_matrix(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object,'F1 score of XGBoost::' + str(
                    self.xgboost_f1_score)+ '\t' +'confiux matrix  for xgboost tree ::' + str(self.xgb_confiux_matrix))

            self.dTree=self.get_best_params_for_dessionTree(train_x,train_y)
            self.prediction_dTree=self.dTree.predict(test_x) # prediction using the SVM Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.dTree_auc_score = roc_auc_score(test_y,self.prediction_dTree)
                self.logger_object.log(self.file_object, 'ROC for SVM:' + str(self.dTree_auc_score))
            else:
                self.dTree_f1_score = f1_score(test_y, self.prediction_dTree,average='micro') # F1 score 
                self.dTree_confiux_matrix =confusion_matrix(test_y, self.prediction_dTree) 
                self.logger_object.log(self.file_object,'F1 score for SVM::' + str(
                    self.dTree_f1_score)+ '\t' +'confiux matrix  for dession tree ::' + str(self.dTree_confiux_matrix))
    

            #comparing the two models
            if(self.dTree_f1_score):
                return 'DTree',self.dTree
            

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()