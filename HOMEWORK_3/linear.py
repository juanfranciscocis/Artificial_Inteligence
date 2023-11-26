
# Import libraries
import numpy as np
import matplotlib.pyplot as plt


# Sigmoid function activation function
sigmoid = lambda x: 1 / (1 + np.exp(-x))

# Perceptron class
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100, weights=None):
        self.weights = weights
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.accuracies = []  # List to store accuracies for each epoch

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return sigmoid(summation)

    def calculate_accuracy(self, predictions, labels):
        thresholded_predictions = (predictions >= 0.5).astype(int)
        correct_predictions = np.sum(thresholded_predictions == labels)
        total_predictions = len(labels)
        accuracy = correct_predictions / total_predictions
        accuracy = accuracy/100
        return accuracy


    def train(self, training_inputs, labels):
        # Iterates through each data point in the training set one at a time for a fixed number of epochs.
        for epoch in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * prediction * (1 - prediction) * inputs
                self.weights[0] += self.learning_rate * error * prediction * (1 - prediction)


            predictions = self.predict(training_inputs)
            accuracy = self.calculate_accuracy(predictions, labels)
            self.accuracies.append(accuracy)
            
            print(f'Epoch {epoch + 1}/{self.epochs}, Accuracy: {accuracy}')

            





    


class PerceptronBatch:
    def __init__(self, learning_rate=0.01, epochs=100, weights=None):
        self.weights = weights
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.accuracies = [] 

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return sigmoid(summation)

    def calculate_accuracy(self, predictions, labels):
        thresholded_predictions = (predictions >= 0.5).astype(int)
        correct_predictions = np.sum(thresholded_predictions == labels)
        total_predictions = len(labels)
        accuracy = correct_predictions / total_predictions
        return accuracy



    def train(self, training_inputs, labels):
        # Calculates predictions for all training inputs in one go and then updates weights based on the aggregated errors for the entire dataset
        for epoch in range(self.epochs):
            predictions = self.predict(training_inputs)
            errors = labels - predictions
            self.weights[1:] += self.learning_rate * np.dot(errors * predictions * (1 - predictions), training_inputs)
            self.weights[0] += self.learning_rate * np.sum(errors * predictions * (1 - predictions))

            accuracy = self.calculate_accuracy(predictions, labels)
            self.accuracies.append(accuracy)

            print(f'Epoch {epoch + 1}/{self.epochs}, Accuracy: {accuracy}')



# Plot the decision boundary
def plot_decision_boundary(perceptron, training_inputs, labels):
    if perceptron.weights is None:
        print("Perceptron has not been trained.")
        return

    plt.scatter(training_inputs[:, 0], training_inputs[:, 1], c=labels, cmap=plt.cm.Spectral)

    # Plot the decision boundary
    x_min, x_max = training_inputs[:, 0].min() - 1, training_inputs[:, 0].max() + 1
    y_min, y_max = training_inputs[:, 1].min() - 1, training_inputs[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = np.round(np.array([perceptron.predict(np.array([x, y])) for x, y in np.c_[xx.ravel(), yy.ravel()]]))
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.3)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Decision Boundary')
    plt.show()


