# Covid-19-Sentiment-Analysis-using-word-embedding

In this project, we will be using a Covid 19 tweet dataset with sentiment labels. Based on this dataset we will build a classification model that will classify the tweets based on the sentiment labels.

# The Dataset

The dataset has 4 sentiment labels: 'Fear', 'Anger', 'Joy', 'Sad’. The value count of each sentiment is as follows:

| Sentiment | Count |
| :---: | :---: |
| Fear | 801 |
| Sad |	795 |
| Joy |	727 |
| Anger | 767 |

# Building the Model to classify the Tweets 

There are broadly 14 steps involved in building the classification model:

1.	Initial Setup: In this step, we import all the required libraries required for our classification model. 

    Since we are using Google Colab, we need to mount our drive which contains the input dataset and load the dataset into a dataframe for manipulating the dataset. 

    As an initial exploration of the dataset, we list out the unique values in each column.


2.	Feature Set Generation: In this step, we generate features of interest from the given dataset. The generated features include: 
    Number of Characters, Number of words, Number of capital characters, Number of capital words, Number of punctuations, Number of words in quotes, Number of sentences, Number     of unique words, Number of hashtags, Number of mentions, Count of stopwords, Average word length, average sentence length, Unique words vs word count feature, Stopwords         count vs words counts feature.


3.	Data Preprocessing: As a first step, we remove the user mentions in the tweets and store the tweet into a new column called ‘Tweet’. All the preprocessing will be now done       one this new column. In the next step, we remove the URLs from the tweet, special characters, numbers and punctuations.

    We create a new dataframe to store only the features that we are interested in. This dataframe is called df_dataset. We store the Tweet and sentiment columns into this           dataframe. 

    Now we convert the sentiment values into numerical values and replace the textual values with these numerical ones.


4.	Load Fasttext vectors: Here we load the pretrained fasttext word vectors. A function is used to count the number of vectors available in the loaded word embedding files.         There are 2000000-word vectors in the file. This is the most time-consuming step while executing.


5.	Text Length Distribution: In this step we find the text length distribution and load it as a new feature to our dataframe, to help us in the padding stage to make sentences     of equal length.


6.	Cleaning the tweets: In this stage, we make sure the dataset we’ll be working on is clean before proceeding to further steps.


7.	Tokenize: In this step, we tokenize the tweets given to us and build a dictionary out of them. The dictionary contains 793 tokens.


8.	Padding: The given dataset has input sentences of varying lengths. To perform the classification, we need to have an input of uniform size. To achieve this, we pad the           sentences with 0s.


9.	Embedding Matrix: Here we create an embedding matrix using the input and word embedding vectors that we had loaded earlier. The words with null embeddings are found to be 35     in number. We also have words which that cannot be found, these are from hashtags


10.	Train-Test split: Here we split the dataframe into train and test set following the 80-20 rule. X_train contains 80 rows and X_test has 20 rows from the dataframe.


11.	Creating the model: We create a model using the Keras library. Here we define the model, its layers, parameters and metrics. To evaluate the model, we’ll be using 4 metrics:     Accuracy, F1 Score, Recall, Precision. Since Keras only provides the accuracy for the model, other metrics must be calculated separately.


12.	Training the model: Now we are ready to train the model with the input data and achieve our classification task. The number of epochs is set to 50.


13.	Testing model: Once the training is completed, we can test the model with custom inputs and see the predictions made by the model.

14.	Save the model: We can save the model to use it in the future without performing all the preprocessing and training.

