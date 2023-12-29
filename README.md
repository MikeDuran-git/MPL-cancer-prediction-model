# Cancer Classification with Multi-Layer Perceptron

## Overview
This project involves the development of a Multi-Layer Perceptron (MLP) model for classifying cancer based on various features. The model is implemented from scratch using Python and tested on a dataset for its accuracy and performance.

## Dataset
The dataset used in this project is a collection of cancer data that includes various features related to cancer attributes. The data is processed and normalized to ensure efficient training of the neural network.

## Model Architecture
The MLP model implemented in this project has the following architecture:

- **Input Layer**: Consists of 30 neurons, corresponding to the number of features in the dataset.
- **Hidden Layer**: A single hidden layer with 15 neurons.
- **Output Layer**: Comprises 2 neurons for binary classification (cancer or no cancer).

Activation functions used include Sigmoid, Tanh, and ReLU. The model utilizes the cross-entropy cost function for error calculation.

## Training the Model
The MLP model is trained using the standard backpropagation algorithm. Training involves feeding forward the inputs, calculating the loss, and updating the weights and biases through backpropagation.

## Evaluation
Post-training, the model is evaluated on a separate test dataset to calculate its accuracy. The accuracy metric provides an indication of how well the model performs in classifying cancer correctly.

## Results
After training, the model achieves an accuracy of 95% on the test dataset. This high accuracy indicates that the model is effective in classifying cancer based on the given features.

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
