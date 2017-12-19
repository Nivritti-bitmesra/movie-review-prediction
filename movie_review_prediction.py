import re
from collections import Counter
from nltk.corpus import stopwords
from os import listdir
import numpy as np
from numpy import array
from sklearn.naive_bayes import MultinomialNB
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten , Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import EarlyStopping , ReduceLROnPlateau
from sklearn import metrics
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

#loading the file
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

#cleaning the file with necessary pre-processing
def clean_doc(doc):
    tokens = doc.split()
    tokens = [re.sub('[^a-zA-Z]',' ', word) for word in tokens]
    tokens = [word.lower() for word in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens
 
#adding words to vocabulary
def add_doc_to_vocab(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)

# taking only the train data into vocabulary
def process_docs(directory, vocab, is_train):
	for filename in listdir(directory):
		if is_train and filename.startswith('cv9'):
			continue
		if not is_train and not filename.startswith('cv9'):
			continue
		path = directory + '/' + filename
		add_doc_to_vocab(path, vocab)
 
vocab = Counter()
process_docs('txt_sentoken/neg', vocab, True)
process_docs('txt_sentoken/pos', vocab, True)
print(len(vocab))
print(vocab.most_common(50)) #print 50 most common words and their counts

# removing words having counts less than 2
min_occurance = 2
tokens = [k for k,c in vocab.items() if c >= min_occurance]
print(len(tokens))

#saving vocabulary into a text file.
def save_list(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()
 
vocab_filename = 'vocab.txt'
save_list(tokens, vocab_filename)

vocab = load_doc(vocab_filename)   #loading the saved file.
vocab = vocab.split()
vocab = set(vocab)
 
def clean_doc_load(doc, vocab):   #cleaning the loaded file
    tokens = doc.split()
    tokens = [re.sub('[^a-zA-Z]',' ', word) for word in tokens]
    tokens = [word.lower() for word in tokens]
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens
    

def process_doc_load(directory, vocab, is_train):  
	documents = list()
	for filename in listdir(directory):
		if is_train and filename.startswith('cv9'):
			continue
		if not is_train and not filename.startswith('cv9'):
			continue
		path = directory + '/' + filename
		doc = load_doc(path)
		tokens = clean_doc_load(doc, vocab)
		documents.append(tokens)
	return documents
 
#splitting the data into train and test
positive_docs = process_doc_load('txt_sentoken/pos', vocab, True)
negative_docs = process_doc_load('txt_sentoken/neg', vocab, True)
train_docs = negative_docs + positive_docs

positive_docs = process_doc_load('txt_sentoken/pos', vocab, False)
negative_docs = process_doc_load('txt_sentoken/neg', vocab, False)
test_docs = negative_docs + positive_docs

y_train = array([0 for _ in range(900)] + [1 for _ in range(900)])
y_test = array([0 for _ in range(100)] + [1 for _ in range(100)])

# using keras tokenizer to transform our data into input for machine learning algorithms.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_docs)

def NB_model(train_docs,test_docs,y_train,y_test):
    X_train = tokenizer.texts_to_matrix(train_docs, mode='freq')
    X_test = tokenizer.texts_to_matrix(test_docs, mode='freq')
    
    model = MultinomialNB()
    model.fit(X_train,y_train)  #implement Multinomial NB algo
    
    acc = metrics.accuracy_score(y_test,model.predict(X_test))
    return acc  

def CNN_model(train_docs,test_docs):
    encoded_docs = tokenizer.texts_to_sequences(train_docs)
    max_length = max([len(s.split()) for s in train_docs])
    X_train = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    
    encoded_docs = tokenizer.texts_to_sequences(test_docs)
    X_test = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    
    vocab_size = len(tokenizer.word_index) + 1
    
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length = max_length))
    model.add(Conv1D(filters = 32, kernel_size = 8, activation='relu'))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    return model,X_train,X_test

def ANN_model(train_docs,test_docs):
    X_train = tokenizer.texts_to_matrix(train_docs, mode='freq')
    X_test = tokenizer.texts_to_matrix(test_docs, mode='freq')
    n_words = X_test.shape[1]
    
    model = Sequential()
    model.add(Dense(50, input_shape=(n_words,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    return model,X_train,X_test

def fit(model,X_train,X_test,y_train,y_test,epochs):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    learning_rate_reduction = ReduceLROnPlateau(monitor='acc', patience=3, verbose=1,factor=0.5, 
                                                min_lr=0.00001)
    callbacks = [EarlyStopping(monitor="loss", min_delta=0 , patience=3 , verbose=0 , mode='auto') ,
                 learning_rate_reduction]
    
    model.fit(X_train, y_train, epochs = epochs, callbacks = callbacks)
    
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    
    return acc
    
#fitting Multinomial Naive Bayes algorithm
acc_nb = NB_model(train_docs,test_docs,y_train,y_test)
print('Test Accuracy of Multinomial Naive Bayes : %f' % (acc_nb*100))

#Convolutional Neural Network model
model_CNN,X_train,X_test = CNN_model(train_docs,test_docs)

epochs = 10
acc_conv = fit(model_CNN,X_train,X_test,y_train,y_test,epochs)
print('Test Accuracy of CNN : %f' % (acc_conv*100))


#Artificial Neural Network model
model_ANN,X_train,X_test = ANN_model(train_docs,test_docs)

epochs = 50
acc_ann = fit(model_ANN,X_train,X_test,y_train,y_test,epochs)
print('Test Accuracy of ANN : %f' % (acc_ann*100))


# tranforming data for input to the word2vec model
total_docs = train_docs + test_docs
sentences = [sentence.split() for sentence in total_docs]
print('Total training sentences: %d' % len(sentences))

# defining the work2vec model
model = Word2Vec(sentences, size=100, window=5, workers=8, min_count=1)

words = list(model.wv.vocab)
print('Vocabulary size: %d' % len(words))
 
filename = 'embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)

# reducing the dimension of the data for plotting
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X[:100])  #taking only the first 100 words

# visualising the word embeddings
plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words[:100]):    #taking only the first 100 words
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()
