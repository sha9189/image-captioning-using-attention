# This file executes all the code that is repetitive to make notebooks more legible

import tensorflow as tf
from sklearn.utils import shuffle
import os

# enable memory growth to use GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# to load and preprocess the image input for InceptionV3 pretrained model
def load_image(image_path):
    """loads and preprocesses image for imception-v3 model
    input:
        image_path ::= string
    returns:
        img ::= Image tensor of shape (299, 299)
        image_path := string
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels = 3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


# create a Inception-V3 model object used for featufre extraction of images

image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

new_input = image_model.input
hidden_layer = image_model.layers[-1].output

def image_features_extract_model():
    return tf.keras.Model(new_input, hidden_layer)


# Loads the Flickr30k Dataset
def load_flickr30k():
    """Loads the Flickr30k Dataset and returns 4 lists in the order of training image name, train captions, validation image names, validation captions.
    Outputs:
        img_name_train
        train_captions
        img_name_val
        val_captions
    """
    # Read the file containing all caption-image pairs
    with open('dataset/flickr30k/Flickr30k.token.txt', 'r') as file:
        annotations = file.read()

    # to load the predefined list of image identfiers for training and validation set
    def load_set(filename):
        """loads the set of identifiers in `filename`"""
        # read the file contents
        with open(filename, 'r') as file:
            doc = file.read()
        dataset = list()
        # process line by line
        for line in doc.split('\n'):
            # skip empty lines
            if len(line) < 1:
                continue
            # get the image identifier
            # identifier = line.split('.')[0]
            dataset.append(line)
        return set(dataset)

    # load the train set identifiers
    train_set = load_set('dataset/flickr30k/Flickr_30k.trainImages.txt')

    # load the validation set identifiers
    val_set = load_set('dataset/flickr30k/Flickr_30k.devImages.txt')

    print("Number of distinct images in training set:", len(train_set))
    print("Number of distinct images in validation set:", len(val_set))


    # get path to the image folder
    PATH = os.path.abspath('.') + '/dataset/flickr30k/flickr30k-images/'

    # Store captions and image names in vectors
    train_captions = []
    img_name_train = []
    val_captions = []
    img_name_val = []

    # splitting the file contents by line
    for annot in annotations.split("\n"):
            # Skip empty lines
            if len(annot)<1:
                continue
            # separate out the caption from the line
            caption = annot.split()[1:]
            # add <start> and <end> token to the caption
            caption = "<start> " + ' '.join(caption) + " <end>"
            # separate out the image id from line)
            image_id = annot.split()[0]
            # remove caption number
            image_id = image_id.split('#')[0]
            # convert image id into the image path
            full_image_path = PATH + image_id

            # add the image id and caption in the repective lists
            if image_id in train_set:
                train_captions.append(caption)
                img_name_train.append(full_image_path)
            elif image_id in val_set:
                val_captions.append(caption)
                img_name_val.append(full_image_path)
    # Shuffle captions and image names together
    train_captions, img_name_train = shuffle(train_captions, img_name_train, random_state = 1)
    print("Length of (img_name_train, train_captions): ({}, {})".format(len(img_name_train), len(train_captions)))
    print("Length of (img_name_val, val_captions): ({}, {})".format(len(img_name_val), len(val_captions)))
    return img_name_train, train_captions, img_name_val, val_captions

