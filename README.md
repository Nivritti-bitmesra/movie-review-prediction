# movie-review-prediction
# Predicting whether the review is positive or negative , comparing performance of various models and creating word embeddings using word2vec model. 

The dataset consists of 1000 positive reviews and 1000 negative reviews made into two different folders.
The dataset can be downloaded from the following link :
http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz

First of all, data is loaded from the two folders. The data is cleaned using ntlk stopwords and by removing punctuations. A vocabulary of words is created using only the words we are keeping aside for training our model. 
After creating the vocabulary , we load the training data and the test data based off the vocabulary that we created.

Necessary pre-processing of the text is done before loading into the different machine learning models using Keras Tokeniser API.

The models used for prediction are :
1. Multinomial Naive Bayes 
2. Convolutional Neural Network 
3. Artificial Neural Network

After that we create word embeddings using the word2vec model. We reduce the dimensionality of the embeddings using PCA so that we are able to visualise the embeddings on a 2-dimensional plot.

