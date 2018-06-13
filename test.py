import Network

net = Network([2, 3, 1])

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)