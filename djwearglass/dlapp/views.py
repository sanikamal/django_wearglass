import os, datetime, random

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.shortcuts import render
import tensorflow as tf
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import model_from_yaml
from tensorflow.keras.models import model_from_json
import numpy as np
from PIL import Image
from djwearglass.settings import MEDIA_ROOT
# from django.conf import settings

def handle_uploaded_file(f):
    name = str(datetime.datetime.now().strftime('%H%M%S')) + str(random.randint(0, 1000)) + str(f)
    path = default_storage.save(MEDIA_ROOT + '/' + name,
                                ContentFile(f.read()))
    return os.path.join(MEDIA_ROOT, path), name

def index(request):
    prediction=''
    if request.POST:
        module_dir = os.path.dirname(__file__) 
        file1_path, file1_name = handle_uploaded_file(request.FILES['file1'])
        json_file = open(os.path.join(module_dir,'model.json'))
        model_file =os.path.join(module_dir,'model.h5')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(model_file)   
        # img = image.load_img(file1_path, target_size=(150,150))
        # x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)
        image = Image.open(file1_path)
        image = image.resize((150,150),Image.ANTIALIAS)
        nd_array = np.array(image)
        nd_array_lst = convert_to_ndarry([nd_array,])
        preds = loaded_model.predict(nd_array_lst)
        if preds[0]>0:
            # print(preds + " is  wearglass")
            prediction="Wear Eye glass"
        else:
            # print(preds + " is notwearglass")
            prediction="Not Wear Eye glass"
        return render(request, "index.html", {"prediction":prediction,
                                              "post": True,
                                              "img1src": file1_name,
                                              })
    return render(request, "index.html", {'post': False})



def convert_to_ndarry(cur_list):
    """Takes a list of ndarrys for images, and converts to a len(list) dimensional ndarray."""
    p = np.expand_dims(cur_list[0],0)
    print("Shape " , p.shape)
    for i in range(len(cur_list)):
        nd_item = cur_list[i]
        if (i>0):
            p = np.insert(p,-1,cur_list[i],0)
    return p