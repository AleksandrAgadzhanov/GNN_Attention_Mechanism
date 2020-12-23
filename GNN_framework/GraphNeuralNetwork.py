import torch.nn as nn
from plnn.modules import Flatten


class GraphNeuralNetwork:

    # When the GNN is initialised, a graph is created which has the same structure as the input neural network but each
    # node now containing the corresponding embedding vector
    def __init__(self, neural_network, input_size, input_embedding_size, hidden_embedding_size, output_embedding_size):
        pass


model = nn.Sequential(
    nn.Conv2d(3, 8, 4, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(8, 16, 4, stride=2, padding=1),
    nn.ReLU(),
    Flatten(),
    nn.Linear(16 * 8 * 8, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

for child in model.children():
    print(child)
    for node in child.parameters():
        print(node.size())
