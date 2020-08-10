This folder contains the pretrained glove word vectors downloaded from [this site](https://nlp.stanford.edu/projects/glove/). I used the 42-billion-token 300-dimensional word vectors and extracted out the vectors for words in my model's vocabulary. These vectors were used to build the embedding matrix(which can be non-trainable as these embeddings are pretrained). This trick reduces the learnable weights by 8000(vocab_size) * 300(embedding dimension) = 2.4M and hence dignificantly reduces the training time of the model. This allows for more training and enables quick experimentations.

You can find the complete set of related opeartions in the notebook "03-miscellaneous.ipynb" under the heading "Prepare pretrained embedding matrix".

Happy learning! 
