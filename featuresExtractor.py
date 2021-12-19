import os
from pickle import dump

import numpy as np
from PIL import Image
from keras.applications.inception_v3 import InceptionV3
# small library for seeing the progress of loops.
from tqdm.notebook import tqdm as tqdm

tqdm().pandas()


def extract_features(directory):
    model = InceptionV3(include_top=False, pooling='avg')
    features = {}
    for img in tqdm(os.listdir(directory)):
        filename = directory + "/" + img
        print("image: ", img)
        image = Image.open(filename, 'r')
        image = image.convert('RGB')
        image = image.resize((299, 299))
        image = np.expand_dims(image, axis=0)
        # image = preprocess_input(image)
        image = image / 127.5
        image = image - 1.0
        feature = model.predict(image)
        features[img] = feature
    return features


# 2048 feature vector
features = extract_features('/Users/regel/fiftyone/coco-2017/validation/data')
dump(features, open("features.p", "wb"))
