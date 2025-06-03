import azure.functions as func
import datetime
import json
import logging
import cv2
import matplotlib.pyplot as plt
import asyncio
import numpy as np

#functions from the processing pipeline
from edge_new_params import process_main_pipeline, draw_region, nonmax

app = func.FunctionApp()

#FUNCTION TO PROCESS THE IMAGE AND RETURN COUNT
@app.route(route="count_items", auth_level=func.AuthLevel.ANONYMOUS)                       
def count_items(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('PROCESSING THE IMAGE')

    #empty list
    detections = []

    #call processing pipeline
    try:
        image_files = req.files.getlist("image_upload")
        #if uploaded then fine or else throw error
        if not image_files:
            return func.HttpResponse(f"No file found", status_code=400) #bad request
        else:
            for file in image_files:
                fn = file.filename
            #file -> buffer -> cv2 
                image_file = file.stream.read()
                image_np = np.frombuffer(image_file, np.uint8)
                image_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR) #should be RGB channel

                det = process_main_pipeline(image_cv2)
                if det > 0:
                    logging.info(f"type of det: {type(det)}")
                    detections.append((fn,det))
                else:
                    logging.info(f"! EMPTY TRAY !")
                    detections.append((fn,"!EMPTY TRAY!"))
                logging.info(f"Total: {detections}")

            #might also return the draw regions thing later for display 
            return func.HttpResponse(str(detections), status_code=200) #successful
    
    except Exception as err:
        logging.info(f"Image could not be processed. ERROR: {err}")
        return func.HttpResponse(str(err), status_code=400) #bad request



    
