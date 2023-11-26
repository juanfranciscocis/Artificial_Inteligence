
import numpy as np
import matplotlib.pyplot as plt

def activate(x):
    return 1 if x > 0 else 0

class PerceptronXOR:
    def __init__(self, learning_rate=0.01,epochs=100, weights=None):
        self.weights = weights
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.accuracies = []

    def predict(self, inputs):
        sumation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return activate(sumation)

    def calculate_accuracy(self, predictions, labels):
        thresholded_predictions = [1 if prediction > 0.5 else 0 for prediction in predictions]
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

            predictions = [self.predict(inputs) for inputs in training_inputs]
            accuracy = self.calculate_accuracy(predictions, labels)
            self.accuracies.append(accuracy)
            
            print(f'Epoch {epoch + 1}/{self.epochs}, Accuracy: {accuracy}')





class PerceptronBatchXOR:
    def __init__(self,learning_rate=0.01, epochs=100, weights=None):
        self.weights = weights
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.accuracies = []

    def predict(self, inputs):
        sumation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return activate(sumation)

    def calculate_accuracy(self, predictions, labels):
        thresholded_predictions = [1 if prediction > 0.5 else 0 for prediction in predictions]
        correct_predictions = np.sum(thresholded_predictions == labels)
        total_predictions = len(labels)
        accuracy = correct_predictions / total_predictions
        accuracy = accuracy/100
        return accuracy

    def train(self, training_inputs, labels):
        # Calculates predictions for all training inputs in one go and then updates weights based on the aggregated errors for the entire dataset
        for epoch in range(self.epochs):
            predictions = np.array([self.predict(inputs) for inputs in training_inputs])
            errors = np.array([label - prediction for prediction, label in zip(predictions, labels)])
            errors = errors.reshape(-1, 1)  # Reshape to (100, 1)

            self.weights[1:] += self.learning_rate * np.dot(training_inputs.T, errors).flatten()
            self.weights[0] += self.learning_rate * errors.sum()

            accuracy = self.calculate_accuracy(predictions, labels)
            self.accuracies.append(accuracy)

            print(f'Epoch {epoch + 1}/{self.epochs}, Accuracy: {accuracy}')


def plot_decision_boundary(model, X, y):
    h = .02  # Tamaño del paso en la malla
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = np.array([model.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)



