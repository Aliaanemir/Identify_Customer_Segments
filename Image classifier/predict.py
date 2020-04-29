import tensorflow as tf
import tensorflow_hub as hub
import argparse
from PIL import Image
import numpy as np
import json

#processing the terminal input
parser = argparse.ArgumentParser(description='This is an image classifier.')

parser.add_argument('image_path', action="store")
parser.add_argument('model', action="store")
parser.add_argument('--top_k', action="store", dest="top_k", type = int)
parser.add_argument('--category_names', action="store", dest="category_names")

result = parser.parse_args()

image_path = result.image_path
model = result.model
top_k = result.top_k
category_names = result.category_names

#loading the moel
reloaded_SavedModel = tf.keras.models.load_model('./my_saved_model.h5', custom_objects={'KerasLayer':hub.KerasLayer})

#image prediction and processing
def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224)) 
    image /= 255
    return image.numpy()

if top_k == None:
    top_k = 1

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    prepared_image = np.expand_dims(processed_test_image, axis = 0)
    probs = reloaded_SavedModel.predict(prepared_image)
    probs_new = probs
    returned_probs = []
    returned_classes = []
    for i in range(top_k):
        max_class = np.argmax(probs_new)
        returned_classes.append(max_class)
        returned_probs.append(probs[0,max_class])
        probs_new = np.delete(probs_new, max_class) 
    return returned_probs, returned_classes

image = np.asarray(Image.open(image_path)).squeeze()
probs, classes = predict(image_path, model, top_k)

#mapping to class name
if(category_names):
    with open(category_names, 'r') as f:
        class_names = json.load(f)
    keys = [str(x+1) for x in list(classes)]
    classes = [class_names.get(key) for key in keys]
    #printing the final result
    for i in np.arange(top_k):
        print('Class: {}'.format(classes[i]))
        print('Probability: {:.3%}'.format(probs[i]))
        print('\n')
else:
    print('The top classes are {} with probabilities {}.'.format(classes, probs))
