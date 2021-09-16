from wsgiref import simple_server
from application_logging.logger import App_Logger
from flask import Flask, request, render_template,send_file
from flask import Response
import os
from prediction_validation_insert import Predication_validation
from trainModel import TrainModel
from training_validation_insertion import train_validation
from predictFromModel import prediction
import json
import shutil

import pandas as pd
app = Flask(__name__)

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')



@app.route("/", methods=['GET'])

def home():
    return render_template('base.html')



@app.route("/predictBatch/",methods=['POST'])

def predictBatchRoute():
    try:
        if request.method == 'POST':
            #batchpath = request.form['batchpath']
            print(request.files)
            cwd = os.getcwd()
            try:
                if 'file' in request.files:
                    batch_file = request.files['file']
                    if os.path.exists('predication_Batch_File'):
                        file = os.listdir('predication_Batch_File')
                        if not len(file) == 0:
                            os.remove('predication_Batch_File/' + file[0])
                    else:
                        pass

                    if os.path.exists('Prediction_Database'):
                        file = os.listdir('Prediction_Database')
                        if not len(file) == 0:
                            os.remove('Prediction_Database/' + file[0])
                    else:
                        pass

                    if os.path.exists('Prediction_Logs'):
                        file = os.listdir('Prediction_Logs')
                        if not len(file) == 0:
                            for f in file:
                                os.remove('Prediction_Logs/' + f)
                    else:
                        pass

                    if os.path.exists('Prediction_Output_File'):
                        file = os.listdir('Prediction_Output_File')
                        if not len(file) == 0:
                            os.remove('Prediction_Output_File/' + file[0])
                    else:
                        pass

                    if os.path.exists('Prediction_Raw_files_validated/Bad_Raw'):
                        file = os.listdir('Prediction_Raw_files_validated/Bad_Raw')
                        if not len(file) == 0:
                            os.remove('Prediction_Raw_files_validated/Bad_Raw/' + file[0])
                    else:
                        pass

                    

                    if os.path.exists('PredictionArchiveBadData'):
                        shutil.rmtree('PredictionArchiveBadData')
                    else:
                        pass

                   

                    if os.path.exists('PredictionFileFromDB'):
                        file = os.listdir('PredictionFileFromDB')
                        if not len(file) == 0:
                            os.remove('PredictionFileFromDB/' + file[0])
                    else:
                        pass

                    

                    batch_file.save('predication_Batch_File/' + batch_file.filename)
                    print('Uploaded Successfully !!')
            except Exception as e:
                print(e)

            except OSError as o:
                print(str(o))

            MainFilePath = 'predication_Batch_File' + '/' + batch_file.filename
            

            prediction_val = Predication_validation('predication_Batch_File')
            prediction_val.predication_validation()

            pred = prediction(MainFilePath)
            path = pred.predictionFromModel()
            
            return Response("Prediction File Created at !! " + str(path))
            return Response('Done')
        else:
            print('None Request Matched')
    except ValueError:
        return Response('Error Occured! %s' % str(ValueError))
    except KeyError:
        return Response('Error Occured! %s' % str(KeyError))
    except Exception as e:
        return Response('Error Occured! %s' % str(e))
@app.route("/predict/", methods=['POST'])

def predictRouteClient():
    try:
        if request.json is not None:
            path = request.json['filepath']

            pred_val = Predication_validation(path) #object initialization

            pred_val.predication_validation() #calling the prediction_validation function

            pred = prediction(path) #object initialization

            # predicting for dataset present in database
            path = pred.predictionFromModel()
            return Response("Prediction File created at %s!!!" % path)
    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)

@app.route("/download/",methods=['GET','POST'])
def download_prediction():
    try:
        file = os.listdir('Prediction_Output_File')
        file = str(file[0])
        return send_file('Prediction_Output_File/' + file,as_attachment=True)
    except Exception as e:
        print(str(e))



@app.route("/train/", methods=['POST'])
def trainRouteClient():

    try:
        if request.json is not None:
            path = request.json['filepath']
            train_valObj = train_validation(path) #object initialization

            train_valObj.train_validation()#calling the training_validation function


            trainModelObj = TrainModel() #object initialization
            trainModelObj.trainingModel() #training the model for the files in the table


    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)



if __name__ =="__main__":
    app.run()