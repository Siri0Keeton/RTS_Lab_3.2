import numpy as np


class Perceptron(object):

    def __init__(self, n, thrsh=4, l_rate=0.1):
        self.thrsh = thrsh
        self.l_rate = l_rate
        self.weights = np.zeros(n + 1)

    def predict(self, inputs):
        summ = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return summ.all()

    def train(self, inputs, labels):
        for i in range(self.thrsh):
            for inputs, lbl in zip(inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.l_rate * (lbl - prediction) * inputs
                self.weights[0] += self.l_rate * (lbl - prediction)


if __name__ == '__main__':

    # Basic setups
    A = [0, 6]
    B = [1, 5]
    C = [3, 3]
    D = [2, 4]

    speeds = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]

    time_deadline = [0.5, 1, 2, 5]

    iter_deadline = [100, 200, 500, 1000]

    threshold = 4

    training_inputs = [np.array(A), np.array(B), np.array(C), np.array(D)]

    label = np.array([1, 0, 0, 0])

    perc = Perceptron(2, l_rate=speeds[4])
    perc.train(training_inputs, label)


    print(perc.predict(np.array([3, 4])))
    print(perc.predict(np.array([0, 7])))