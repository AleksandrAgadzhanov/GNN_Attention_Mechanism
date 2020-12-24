import torch
import torch.nn as nn
from torch.nn import functional as f
from plnn.modules import Flatten


class GraphNeuralNetwork:
    """
    This class represents the overall Graph Neural Network and its functionality
    """
    # When the GNN is initialised, a graph is created which has the same structure as the input neural network but each
    # proper node (where ReLU is applied) now containing the corresponding embedding vector filled with zeros.
    # In addition, all the auxiliary neural networks are initialised including the ones which perform the forward and
    # backward update operations and the one which computes the scores of update methods of all input nodes
    def __init__(self, neural_network, input_size, embedding_vector_size, input_feature_size, hidden_feature_size,
                 output_feature_size, auxiliary_hidden_size):
        # Initialise an input embedding vector at each position in the input size
        self.inputs_embeddings = torch.zeros(torch.Size([*input_size, embedding_vector_size]))

        # Initialise the list of embeddings of each ReLU hidden layer
        self.hidden_embeddings = []

        # Initialise the test input to the network for the purpose of determining the sizes at the output of each layer
        test_image = torch.zeros(input_size)

        # Now iterate over the neural network layers, passing the input through one layer in turn, and initialise the
        # embeddings and edges of each layer of nodes containing ReLU activation functions
        layers = list(neural_network.children())
        while len(layers) != 0:

            # Pop the layer at the index 0 to retrieve the next layer and pass the input through it
            layer = layers.pop(0)
            test_image = layer(test_image)

            # Only add the embeddings to the list if the current layer of interest is the ReLU layer
            if type(layer) == torch.nn.ReLU:
                self.hidden_embeddings.append(torch.zeros(torch.Size([*test_image.size(), embedding_vector_size])))

        # Finally, initialise the output embedding vector of the only output node
        self.output_embedding = torch.zeros(embedding_vector_size)

        # Now, initialise all the auxiliary neural networks
        self.forward_input_update_nn = ForwardInputUpdateNN(input_feature_size, auxiliary_hidden_size,
                                                            embedding_vector_size)
        self.forward_hidden_update_nn = ForwardHiddenUpdateNN(hidden_feature_size, auxiliary_hidden_size,
                                                              embedding_vector_size)
        # TODO


class ForwardInputUpdateNN(nn.Module):
    """
    This class represents the neural network which performs the forward update on the input nodes
    """
    def __init__(self, feature_vector_size, hidden_layer_size, embedding_vector_size):
        super(ForwardInputUpdateNN, self).__init__()
        self.linear_1 = nn.Linear(feature_vector_size, hidden_layer_size)
        self.linear_2 = nn.Linear(hidden_layer_size, embedding_vector_size)

    def forward(self, input_feature_vector):
        return self.linear_2(f.relu(self.linear_1(input_feature_vector)))


class ForwardHiddenUpdateNN(nn.Module):
    """
    This class represents the neural network which performs the forward update on the hidden layer nodes
    """
    def __init__(self, feature_vector_size, hidden_layer_size, embedding_vector_size):
        super(ForwardHiddenUpdateNN, self).__init__()

        # Initialise the layers for obtaining information from the local features
        self.linear_1_local = nn.Linear(feature_vector_size, hidden_layer_size)
        self.linear_2_local = nn.Linear(hidden_layer_size, hidden_layer_size)

        # Initialise the layers for obtaining information from the previous neighbour embedding vectors
        self.linear_1_neighbour = nn.Linear(2 * embedding_vector_size, hidden_layer_size)
        self.linear_2_neighbour = nn.Linear(hidden_layer_size, hidden_layer_size)

        # Finally, initialise the layers for combining information from the local features and the previous neighbour
        # embedding vectors
        self.linear_1_combine = nn.Linear(2 * hidden_layer_size, hidden_layer_size)
        self.linear_2_combine = nn.Linear(hidden_layer_size, embedding_vector_size)

    def forward(self, local_feature_vector, transformed_previous_neighbour_embeddings):
        # First, get information from the local feature vector
        local_features_info = self.linear_2_local(f.relu(self.linear_1_local(local_feature_vector)))

        # Second, get information from the transformed previous neighbour embedding vectors
        previous_neighbour_embeddings_info = self.linear_2_neighbour(f.relu(
            self.linear_1_neighbour(transformed_previous_neighbour_embeddings)))

        # Finally, combine the information from the local features and the transformed previous neighbour embedding
        # vectors
        return self.linear_2_combine(f.relu(self.linear_1_combine(torch.cat([local_features_info,
                                                                             previous_neighbour_embeddings_info]))))


model = nn.Sequential(
    nn.Conv2d(3, 8, 4, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(8, 16, 4, stride=2, padding=1),
    nn.ReLU(),
    Flatten(),
    nn.Linear(16 * 8 * 8, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)


