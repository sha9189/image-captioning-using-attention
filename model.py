# This file has all the objects needed to make a prediction carried over from the 
# notebook "04-modelling-and-training.ipynb". To understand the code, please go through the notebook.

from model_utils import *
import pickle
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# enable memory growth to avoid getting error while using InceptionV3 model
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


###########################
# Code to build the model architecture starts here

# class to pass extracted images thru a fully connected network
class CNN_Encoder(tf.keras.Model):
    """Passes the InceptionV3 image features through a fully connected layer with `embedding_dim` units"""
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # define fully connected layer
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x    

class BahdanauAttention(tf.keras.Model):
    """Class to implement attention mechanism"""
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        # fc layer for image features
        self.W1 = tf.keras.layers.Dense(units)
        # fc layer for previous hidden state of RNN
        self.W2 = tf.keras.layers.Dense(units)
        # fc layer to get attention coeffients per 2D space of the image
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        """Inputs:
            features shape: (?, 64, embedding_dim)
            hidden shape: (?, units)
        Outputs:
            context_vector shape: (?, embedding_dim)
            attention_weights: (?, 64, 1)
        """
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # hidden shape: (?, units)
        # hidden_with_time_axis shape: (?, 1, units)

        # combine feature activations with hidden state activations
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        # score shape: (?, 64, units)

        # get the attention weights(coefficients/importance) in the 2D space
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        # attention_weights shape: (?, 64, 1)

        # get the attention weighted image features
        context_vector = attention_weights * features
        # context_vector shape: (?, 64, embedding_dim)

        # consolidate all the activations in the 2D space as one number representing each feature
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # context_vector shape:(?, embedding_dim)

        return context_vector, attention_weights


class RNN_Decoder(tf.keras.Model):
    """Implements RNN for predicting the next word in the sequence"""
    def __init__(self, embedding_dim, units, vocab_size, embedding_matix):
        super(RNN_Decoder, self).__init__()
        self.units = units
        # for getting embedding vector of last word generated
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matix], trainable=False)
        # GRU for processing context_vector after attention
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

        # fc layers after gru
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        # attention object to get context_vector
        self.attention = BahdanauAttention(self.units)

    def __call__(self, x, features, hidden):
        """Inputs:
            x: integer representing last predicted word
            x shape: (?, 1)

            features: image features after passing through encoder
            features shape: (?, 64, embedding_dim)

            hidden: stored last hidden state of GRU
            hidden shape: (?, units)
        Outputs:
            x: prediction for the next word
            x shape: (?, vocab_size)

            state: hidden state of gru
            state shape: (?, units)

            attention_weights shape: (?, 64, 1)
        """
        # get the context vector using attention
        context_vector, attention_weights = self.attention(features, hidden)
        # context_vector shape: (?, embedding_dim)
        
        # get embedding vector of last word in the sequence
        x = self.embedding(x)
        # X shape: (?, 1, embedding_dim)

        # concat last word embedding with the context vector
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # pass the concatenated vector thru gru
        output, state = self.gru(x)
        # output shape:(?, 1, units)
        # state shape: (?, units)

        x = self.fc1(output)
        # x shape: (?, 1, units)

        # remove the extra dimension
        x = tf.reshape(x, (-1, x.shape[2]))
        # x shape: (?, units)

        x = self.fc2(x)
        # x shape: (?, vocab_size)

        return x, state, attention_weights
    
    #reset hidden state to be run before executing a new batch
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


# load the weights for embedding layer
with open('embedding_matix.pkl', 'rb') as file:
    embedding_matix = pickle.load(file)


# Set up the model configuration constants
embedding_dim = 303
units = 512
vocab_size = 8000
attention_features_shape = 64
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size, embedding_matix)

# Create an object for InceptionV3 model
image_features_extract_model = image_features_extract_model()

# Restore the model with with beam score
ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
ckpt.restore('./checkpoints/backup/ckpt-29')

# Code to build the model architecture ends here
#####################################

#####################################
# Model Prediction code starts here

# load the trained tokenizer
# load the tokenizer
with open("tokenizer-8k-vocab.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# max_length is used to limit the length of predicted caption
max_length = 80

# Evaluate Using Beam Search
def evaluate(image, beam_index=3):
    # list to store top_b_seq: [[seq], log(score)]
    top_b_seq = [[[tokenizer.word_index['<start>']], 0.0]]
    # [[[seq1], log(0.0)], [[seq2], log(0.2)]]

    # stores hidden states of parent sequences : {i: (token, hidden)}
    hidden_cache = {}
    # stores hidden states of new predictions
    new_hidden_cache = {}
    # initialize for <start> token
    hidden_cache[0] = (tokenizer.word_index['<start>'], decoder.reset_state(batch_size=1))

    # prepare image to feed into the model
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    # temp_input shape: (1, 299, 299, 3)

    # pass temp_input thru the InceptionV3 model
    img_tensor_val = image_features_extract_model(temp_input)
    # img_tensor_val shape: (1, 8, 8, 2048)

    # prepare img_tensor_val to feed into encoder
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    # img_tensor_val shape: (1, 64, 2048)

    features = encoder(img_tensor_val)
    # features shape: (1, 64, embedding_dim)

    # Loop over for each word(<start> token excluded)
    for var in range(max_length-1):
        # create list to store candidate sequences
        candidate_seq = list()
        # iterate over each parent seq
        for i, p_seq in enumerate(top_b_seq):
            # select the last token as the decoder input
            last_token = p_seq[0][-1]
            # if last_token is <end>, add it as a candidate_seq and move to next p_seq
            if last_token == tokenizer.word_index['<end>']:
                candidate_seq.append(p_seq)
                continue
            # get the decoder input
            dec_input = tf.expand_dims([last_token], 0)
            # find the right hidden state with respect to 2nd last token
            for key in hidden_cache.keys():
                # select zero hidden state for first prediction
                if len(p_seq[0]) < 2 :
                    hidden = hidden_cache[0][1]
                    break
                elif hidden_cache[key][0] == p_seq[0][-2]:
                    hidden = hidden_cache[key][1]
                    break
            # use retrieved hidden state and dec_input for prediction
            predictions, new_hidden, _ = decoder(dec_input, features, hidden)
            # predictions: (1, vocab_size)
            # new_hidden: (1, units)
            # save the hidden state with token used for next word prediction
            new_hidden_cache[i] = (last_token, new_hidden)

            # add a softmax layer for beam search to work
            predictions=tf.nn.softmax(predictions)


            # Get the top probabilties and add them in candidate seqs
            top_probs, top_idxs = tf.math.top_k(predictions[0], k=beam_index)
            for prob, next_token in zip(top_probs, top_idxs):

                new_seq = p_seq[0].copy()
                new_seq.append(next_token.numpy())
                # remove length normalization(alpha=0.7) from stored prob score
                old_prob = p_seq[1] * np.power(len(new_seq)-1, 0.7)
                # add log of new prob and apply length normalization to the total
                total_prob = (old_prob+np.log(prob.numpy())) / np.power(len(new_seq), 0.7)
                
                candidate_seq.append([new_seq, total_prob])
        # After collecting all potential candiadate seq, sort them on the total probability
        candidate_seq = sorted(candidate_seq, reverse = True, key= lambda l: l[1])
        # assign top `beam_index` seqs as top_b_seq
        top_b_seq = candidate_seq[:beam_index]
        
        # replace the hedden_cache with new_hidden_cache values before iterating over next set of parent sequences
        hidden_cache = new_hidden_cache
    # return seq with highest prob
    result = tokenizer.sequences_to_texts([top_b_seq[0][0]])
    return result

# Evaluate using greedy serach
def evaluate_with_attention(image):
    """function to predict the caption given an image path
    Inputs:
        image: full image path
    Outputs:
        result: list sequentially representing the predicted caption
        attention_plot: numpy array representing attention to each 8x8 block of the image per word
        attention_plot shape: (caption length, 64)
    """
    # create an array to capture the attention weights per predicted word
    attention_plot = np.zeros((max_length, attention_features_shape))
    # attention_plot shape: (max_length, 64)

    # reset the hidden state of decoder
    hidden = decoder.reset_state(batch_size=1)
    # hidden shape: (batch_size, units)

    # prepare image to feed into the model
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    # temp_input shape: (1, 299, 299, 3)

    # pass temp_input thru the InceptionV3 model
    img_tensor_val = image_features_extract_model(temp_input)
    # img_tensor_val shape: (1, 8, 8, 2048)

    # prepare img_tensor_val to feed into encoder
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    # img_tensor_val shape: (1, 64, 2048)

    features = encoder(img_tensor_val)
    # features shape: (1, 64, embedding_dim)

    # set up decoder input as the <start> token
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    # dec_input shape: (1, 1)

    # empty list to capture predicted words
    result = []
    
    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        # predictions shape: (1, vocab_size)
        # hidden shape: (1, units)
        # attention_weights shape: (1, 64, 1)

        # store the attention weights to use for the plot later
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        # get the predicted word id
        predicted_id = tf.math.argmax(predictions, axis=1)[0].numpy()
        # predicted: integer

        # find the corresponding word and add in the result list
        result.append(tokenizer.index_word[predicted_id])

        # check for end of sequence condition
        if tokenizer.index_word[predicted_id] == '<end>':
            # cut short the attention_plot array
            attention_plot = attention_plot[:len(result), :]
            return result, attention_plot
        
        # update decoderinput to the predicted word
        dec_input = tf.expand_dims([predicted_id], 0)

    # return the results
    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot 


def plot_attention(image, result, attention_plot):
    """Function to plot the attention graphs for the predicted caption
    Inputs:
        image: full path to the image
imageimage        result: list sequentially representing the predicted caption
        attention_plot: numpy array representing attention to each 8x8 block of the image per word
        attention_plo# set up decoder input as the <start> token
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)t shape: (caption length, 64)
    Outputs: None, but plots the graph
    """
    # open image as an array
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(20, 20))

    len_result = len(result)
    for l in range(len_result):
        # get the attention weights for (l+1)th caption
        temp_att = np.resize(attention_plot[l], (8, 8))
        # temp_att shape:(8,8)

        # print("len_result//2:", len_result//2)
        # print("l+1:", l+1)
        ax = fig.add_subplot(len_result//2, 4, l+1)
        # set title of subplot as the word
        ax.set_title(result[l])
        #display the image in the subplot
        img = ax.imshow(temp_image)
        # overlap attention on image with alpha=0.6
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
    
    plt.tight_layout()
    plt.show()

# Model Prediction code ends here
#######################################