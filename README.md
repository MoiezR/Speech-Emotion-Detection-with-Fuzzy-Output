# Speech-Emotion-Detection-with-Fuzzy-Output
In this project I use the Toronto Emotional Speech Set or the TESS dataset. The model primarily makes use of the Long Short Term Memory or LSTM Unit to be trained, it has a linear activation function. The output depicts the degree of membership of each element in the dataset with regards to each of the seven labels.

The TESS dataset comes with 7 emotion labels: fear, sad, neutral, disgust, pleasantly surprised, angry and happy. All the labels have ample data to train and test our model.

We now use the librosa library to use its feature and extract the MFCC (Mel-frequency cepstral coefficients) feature from our audio files. We take these MFCC features from all of our audio samples and store it in an array. We now convert the array into a 2D one where each array in the set of arrays represents an audio sample and the numbers inside the array represent its MFCC values.

We now prepare this array by reshaping and encoding it (using OneHotEncoder) to prepare it for the input of our neural network model.

We then implement a neural network model using the Keras library. The neural network model is defined using the 'Sequential' class from Keras, indicating a linear stack of layers. The model starts with an LSTM layer with 123 units. The return_sequences=False means it's a final layer in the sequence. The input_shape=(40, 1) specifies the input shape, where 40 is the number of time steps (MFCC features) and 1 is the number of features per time step.Two dense layers follow the LSTM layer. The first has 64 units with ReLU (Rectified Linear Unit) activation, and the second has 32 units with ReLU activation.Dropout layers are inserted after the dense layers with a dropout rate of 0.2. Dropout is a regularization technique that helps prevent overfitting by randomly setting a fraction of input units to zero during training.The final dense layer has 7 units (assuming 7 emotion classes) with linear activation. Linear activation is used for regression tasks.The model is compiled using mean squared error as the loss function, Adam as the optimizer, and mean squared error as the evaluation metric.

We then train the model with the following specifications: 
The fit method is used to train the model on the input data (X) and target labels (y).
validation_split=0.2 specifies that 20% of the training data will be used as a validation set during training.
epochs=100 indicates the number of times the entire dataset is processed by the model during training.
batch_size=512 determines the number of samples used in each iteration of gradient descent.
shuffle=True indicates that the training data will be shuffled at the beginning of each epoch.

After the training and compilation we assign label names to our x axis same as the emotion labels and visually depict the membership value of an audio sample with regards to its classification to each of the labels.
