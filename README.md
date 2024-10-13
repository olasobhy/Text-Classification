# Text Classification with Preprocessing and Deep Learning
Overview
The purpose of this project is to classify text into five distinct categories using natural language processing (NLP) techniques and deep learning. The process involves:

Preprocessing raw text data to remove noise.
Converting the cleaned text into numerical format suitable for the model.
Training a neural network to classify the text data into predefined categories.
Requirements
This project requires several Python libraries, including NumPy, Pandas, Matplotlib, Scikit-learn, Keras, and NLTK. Make sure to install them before running the project. Additionally, stop words need to be downloaded for effective text cleaning.

Data Preprocessing
The text data undergoes several preprocessing steps:

Text Cleaning: Non-alphabetical characters are removed, and the text is converted to lowercase.
Stemming: Words are reduced to their base form (e.g., "running" becomes "run").
Stopword Removal: Commonly used words (e.g., "and", "the") that do not contribute to sentiment or classification are removed.
The preprocessed text is then tokenized and converted into sequences of numbers, which are padded to ensure uniform length across all data points.

Model Architecture
The model architecture includes:

An Embedding Layer to convert words into dense vectors of a fixed size.
A Global Average Pooling Layer to average the sequence of word vectors.
Fully connected Dense Layers to perform the classification.
The final output layer uses a softmax activation function to classify the text into one of the five categories.
Training and Evaluation
The model is trained on the preprocessed text data using the Adam optimizer and sparse categorical crossentropy loss. The training process also tracks validation accuracy to monitor how well the model generalizes to unseen data. The trained model is then evaluated on a separate test set to measure its performance.

Results
The accuracy of the model, both for training and validation, is plotted over the number of epochs. This provides a visual insight into how well the model learns from the data and whether it's overfitting or underfitting.

Saving the Model
The trained model is saved as an HDF5 file (model.h5), allowing it to be reused without retraining. This is useful for deploying the model in a production environment or sharing it with others
