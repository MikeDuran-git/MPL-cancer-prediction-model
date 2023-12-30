# Cancer Classification with Multi-Layer Perceptron

## Overview
This repository contains an implementation of a Multi-Layer Perceptron (MLP) from scratch, designed to classify cancer from diagnostic data. The project demonstrates the application of fundamental neural network concepts using Python.

## Dataset
The MLP is trained and evaluated on a pre-processed cancer dataset, which includes various features and diagnostic indicators normalized for effective neural network training.

## Starting Model Architecture
The MLP model implemented in this project has the following architecture:

- **Input Layer**: Consists of 30 neurons, corresponding to the number of features in the dataset.
- **Hidden Layer**: A single hidden layer with 15 neurons.
- **Output Layer**: Comprises 2 neurons for binary classification (cancer or no cancer).

Activation functions used include Sigmoid, Tanh, and ReLU. The model utilizes the cross-entropy cost function for error calculation.

## Training the Model
The MLP undergoes training through forward and backward propagation with early stopping to prevent overfitting. The training process iteratively adjusts weights and biases based on the calculated loss.

## Evaluation
We tested multiple combinations with varied values for the hidden layer, the activation function and the learning rate.

Evaluation metrics such as accuracy, precision, recall, F1 score, and specificity quantify the model's performance. These metrics are derived from the model's predictions on a holdout test set.

## Results
After training, and evaluating the model and execute multiple simulations we identified the optimal parameters:
```python
params={
        'InputLayer': 30,
        'HiddenLayer': 20,
        'OutputLayer': 2,
        'LearningRate': 0.001,
        'Epocas': 600,
        'ActivationFunction': 'sigmoid'    
    }
```
After training, the model achieves an accuracy of 99.12% on the test dataset. This high accuracy indicates that the model is effective in classifying cancer based on the given features.

## Loss Function Visualization
The loss function over epochs is plotted to visualize the model's learning process. A decreasing loss trend indicates that the model is learning effectively.

## Usage
To use the model:

1. Load the dataset.
2. Preprocess the data (normalize, split into training and testing sets).
3. Create an instance of the `MultiLayerPerceptron` class.
4. Train the model using the `fit` method.
5. Evaluate the model's accuracy on the test set.
6. Visualize the loss function using the `plot_loss` method.

## Conclusion
This MLP model demonstrates the capability of neural networks in classifying complex patterns like cancer indicators. The high accuracy achieved suggests that the model could be a useful tool in medical diagnostics, alongside professional medical advice.

## Future Work
Future enhancements to the model could include:

- Implementing additional layers in the neural network to improve its learning capacity.
- Experimenting with different activation functions and learning rates.
- Applying cross-validation to ensure the model's robustness and reliability.
