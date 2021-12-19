# load descriptions
import json
import string
from pickle import dump

from keras.preprocessing.text import Tokenizer


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def load_descriptions(metadata):
    mapping = dict()
    annotations = metadata['annotations']

    for annotation in annotations:
        image_id = str(annotation['image_id'])
        image_desc = annotation['caption']
        if image_id not in mapping:
            mapping[image_id] = list()
        mapping[image_id].append(image_desc)
    return mapping


def clean_descriptions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = ' '.join(desc)

    return descriptions


# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def load_images(metadata):
    imagesById = dict()
    images = metadata['images']
    for image in images:
        identifier = str(image['id'])
        imagesById[identifier] = image['file_name']
    return imagesById


# load clean descriptions into memory
def map_descriptions_with_images(descriptions, imagesById):
    mappedDescriptions = dict()
    for image_id, image_descs in descriptions.items():
        if image_id in imagesById:
            path = imagesById[image_id]
            if path == '000000130465.jpg':
                continue
            if path not in mappedDescriptions:
                mappedDescriptions[path] = list()
            for desc in image_descs:
                mappedDescriptions[path].append('startseq ' + desc + ' endseq')

    return mappedDescriptions


# converting dictionary to clean list of descriptions
def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


# creating tokenizer class
# this will vectorise text corpus
# each integer will represent token in dictionary
def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer


# calculate maximum length of descriptions
def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)


# load training dataset


metadataJsonFile = open("/Users/regel/fiftyone/coco-2017/raw/captions_val2017.json")
metadata = json.load(metadataJsonFile)
metadataJsonFile.close()

descriptions = load_descriptions(metadata)
descriptions = clean_descriptions(descriptions)
imagesById = load_images(metadata)
imagePathsWithDescriptions = map_descriptions_with_images(descriptions, imagesById)
dump(imagePathsWithDescriptions, open("descriptions.p", "wb"))
print("image count", len(imagePathsWithDescriptions))

# give each word an index, and store that into tokenizer.p pickle file
tokenizer = create_tokenizer(imagePathsWithDescriptions)
dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1  # for to_categorical
print("vocabulary size", vocab_size)

max_length = max_length(descriptions)
print("max length", max_length)
