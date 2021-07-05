#authors : Urvish and Mobassir

import logging
import requests
import os
import io
import glob
import time
import json

import tensorflow as tf
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# redefining transformers custom model

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

# Class map for the sentiment
CLASS_MAP = {0: 'Negative', 1:'Positive'}

##############################################################################
import numpy as np
from moviepy.editor import *

def decode_video(path):
    print("ON EST DANS DECODE VIDEO")
    
    print("avant la methode videofileclip")
    #cap = cv2.VideoCapture(path)
    clip = VideoFileClip(path, target_resolution=(30,30))
    duree = clip.duration
    fps = clip.fps
    nb_frames = fps*duree
    print("durée = ", duree)
    print("fps = ", fps)
    print("nb_frames = ",  nb_frames)
    
    frames = []
    print("AVANT LA BOUCLE FOR")
    mask = np.linspace(nb_frames // 5, (nb_frames // 5) * 4, 8, dtype=np.int)
    print("mask = ", mask)
    for i in mask:
        frames.append(clip.get_frame(i/fps))
    
    """
    for i in range(0, 8):
        frames.append(clip.get_frame(i/2))
        print(str(i))
        """
    '''
    ret = True
    
    frames = []
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            frames.append(img)
    '''
    print("APRES LA BOUCLE FOR")
    video = np.stack(frames, axis=0)
    print(video.shape)
    """
    print("APRES LE STACK")
    try:
        # selection de 8 frames en dehors du premier et dernier quintile
        # video = video[video.shape[0] // 5:(video.shape[0] // 5) * 4, :, :, :]
        mask = np.linspace(video.shape[0] // 5, (video.shape[0] // 5) * 4, 8, dtype=np.int)
        video = np.take(video, mask, axis=0)
    except:
        video = video[:8, :, :, :]"""
    """
    # LARGEUR
    try:
        # selection de 8 frames en dehors du premier et dernier quintile
        # video = video[video.shape[0] // 5:(video.shape[0] // 5) * 4, :, :, :]
        mask = np.linspace(0, video.shape[1], 100, dtype=np.int)
        video = np.take(video, mask, axis=1)
    except:
        video = video[:, :100, :, :]
    
    # LONGUEUR
    try:
        # selection de 8 frames en dehors du premier et dernier quintile
        # video = video[video.shape[0] // 5:(video.shape[0] // 5) * 4, :, :, :]
        mask = np.linspace(0, video.shape[2], 100, dtype=np.int)
        video = np.take(video, mask, axis=2)
    except:
        video = video[:, :, :100, :]
    """
        
    json_obj = json.dumps({'instances': [video.tolist()]})
    print("le json_obj est :")
    print(json_obj)
    print("ON EST SORTI DE DECODE VIDEO")
    return json_obj

####################################################################################

# Deserialize the Invoke request body into an object we can perform prediction on
"""
def input_handler(data, context):
    print("ON EST DANS INPUT HANDLER")
    print(context.request_content_type)
    d = data.read().decode('utf-8')
    print(d)
    # return decode_video(d)
    
    s3_client = boto3.client('s3')

    print("boto3 client est passé")
    # Download the file from S3
    s3_client.download_file('xlmrobertaeddy', d, '1.webm')
    print("ON EST SORTI DE INPUT HANDLER")
    return decode_video("1.webm")
"""

def input_handler(data, context):
    print("ON EST DANS INPUT HANDLER")
    print(context.request_content_type)
    d = data.read().decode('utf-8')
    print(d)

    url = d[1:-1]
    r = requests.get(url, allow_redirects=True)

    with open("toto.webm", 'wb') as f:
        for chunk in r.iter_content(chunk_size = 1024*1024):
            if chunk:
                f.write(chunk)
    
    print("ON EST SORTI DE INPUT HANDLER")
    return decode_video("toto.webm")


#mapping predicted Encoded values to labels
def fun(x, dix):
    return dix[x]

MapEncodedLabel = np.vectorize(fun)

# Serialize the prediction result into the desired response content type
def output_handler(data, context):
    NB_PREDICTION = 5
    #supprimer 1.webm
    print("ON EST DANS OUTPUT HANDLER")
    print(data)
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))
        print("on est dans le if, donc probleme")
    print("après le if status code")
    response_content_type = context.accept_header
    print("response_content_type : ")
    print(response_content_type)
    prediction = json.loads(data.content)
    print("prediction : ")
    print(prediction)
    #preds = np.argmax(np.array(prediction["predictions"]), axis=1)
    preds = np.flip(np.array(prediction["predictions"]).argsort()[0][-NB_PREDICTION:])
    proba = np.array(prediction["predictions"])[0][preds]
    print("preds :")
    print(preds)
    print("proba :")
    print(proba)
    

    #preds = prediction["predictions"][0][0]
    #if (preds >= 0.5):
    #    preds = 1
    #else:
    #    preds = 0
    #idx = MapEncodedLabel(preds, CLASS_MAP).tolist()
    #print(idx)
    json_obj = json.dumps({"predictions" : preds.tolist(),
                          "probabilites" : proba.tolist()})
    print("le json_obj : ")
    print(json_obj)
    print("ON EST SORTIE OUTPUT HANDLER")
    return json_obj, response_content_type
