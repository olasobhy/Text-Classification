# Text Classification with Preprocessing and Deep Learning
## Overview
This project demonstrates how to classify text data into multiple categories using natural language processing (NLP) and deep learning techniques. The key steps include:
Text preprocessing (cleaning, tokenization, stemming)
Converting text into numerical sequences
Building a deep learning model to predict text categories
## Requirements
install the following Python libraries:

NumPy
Pandas
Matplotlib
Scikit-learn
Keras
NLTK (including stopwords)

## Data Preprocessing
Text data is preprocessed by:

Removing non-alphabetical characters.
<br>
Converting text to lowercase.
<br>
Applying stemming to reduce words to their root form.
<br>
Removing stopwords to filter out irrelevant words.
<br>
After preprocessing, the text is tokenized and transformed into numerical sequences. These sequences are padded to ensure uniformity in input length across all samples.
<br>

## Model Architecture
The deep learning model for text classification consists of:
<br>
Embedding Layer: Converts words into dense numerical vectors.
<br>
Global Average Pooling Layer: Reduces the dimensionality by averaging word vectors.
<br>
Dense Layers: Learn patterns from data to perform classification.
<br>
Output Layer: A softmax function to predict one of five categories.
<br>
## Training and Evaluation
The model is trained using the Adam optimizer and sparse categorical crossentropy as the loss function. During training, both accuracy on training data and validation accuracy are monitored. The trained model is evaluated on a test set to assess its performance and it acheive 95%.

## Results
The project includes visualizing the training and validation accuracy over epochs. This plot helps in understanding how well the model is learning and whether there are issues like overfitting or underfitting.

## Saving the Model
After training, the model is saved as model.h5 to allow for reuse in future predictions without retraining. This also facilitates easy sharing of the trained model.

