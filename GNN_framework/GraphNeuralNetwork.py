import torch
import torch.nn as nn
from torch.nn import functional as f
from plnn.modules import Flatten


class GraphNeuralNetwork:
    """
    This class represents the overall Graph Neural Network and its functionality
    """
    # When the GNN is initialised, a graph is created which has the same structure as the input neural network but each
    # proper node (in[ut, ReLU and output) now containing the corresponding embedding vector filled with zeros.
    # In addition, all the auxiliary neural networks are initialised including the ones which perform the forward and
    # backward update operations and the one which computes the scores of update methods of all input nodes
    def __init__(self, neural_network, input_size, embedding_vector_size, input_feature_size, relu_feature_size,
                 output_feature_size, auxiliary_hidden_size, num_update_methods):
        # Store the underlying neural network as a field of the GNN
        self.neural_network = neural_network

        # Initialise a row tensor of tensors each corresponding to a particular pixel
        self.input_embeddings = torch.zeros(torch.Size([*input_size, embedding_vector_size]))
        self.input_embeddings = self.input_embeddings.reshape(-1, embedding_vector_size)

        # Initialise the list of embeddings of each ReLU hidden layer
        self.relu_embeddings = []

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
                relu_embeddings = torch.zeros(torch.Size([*test_image.size(), embedding_vector_size]))
                self.relu_embeddings.append(relu_embeddings.reshape(-1, embedding_vector_size))

        # Finally, initialise the output embedding vectors for the output nodes using the test output from the network
        self.output_embeddings = torch.zeros(torch.Size([*test_image.size(), embedding_vector_size]))
        self.output_embeddings = self.output_embeddings.reshape(-1, embedding_vector_size)

        # Now, initialise all the auxiliary neural networks
        self.forward_input_update_nn = ForwardInputUpdateNN(input_feature_size, auxiliary_hidden_size,
                                                            embedding_vector_size)
        self.forward_hidden_update_nn = ForwardReluUpdateNN(relu_feature_size, auxiliary_hidden_size,
                                                            embedding_vector_size)
        self.forward_output_update_nn = ForwardOutputUpdateNN(output_feature_size, auxiliary_hidden_size,
                                                              embedding_vector_size)
        self.backward_hidden_update_nn = BackwardReluUpdateNN(relu_feature_size, auxiliary_hidden_size,
                                                              embedding_vector_size)
        self.backward_input_update_nn = BackwardInputUpdateNN(input_feature_size, auxiliary_hidden_size,
                                                              embedding_vector_size)
        self.score_computation_nn = ScoreComputationNN(embedding_vector_size, auxiliary_hidden_size, num_update_methods)

    def reset_input_embedding_vectors(self):
        self.input_embeddings = torch.zeros(self.input_embeddings.size())

    # TODO
    def update_embedding_vectors(self, input_feature_vectors, relu_feature_vectors, output_feature_vectors,
                                 num_updates):
        """
        This function performs a series of forward and backward updates on all the embedding vectors until convergence
        is reached
        """
        # Perform one complete forward and backward update a number of times specified
        for i in range(num_updates):

            # First, perform the forward update of the input embedding vectors if they are still all zero (only happens
            # during the first update)
            if torch.eq(self.input_embeddings, torch.zeros(self.input_embeddings.size())).all().item():

                # Perform the forward update on each input embedding vector in turn
                for input_idx in range(self.input_embeddings.size()[0]):
                    self.input_embeddings[input_idx] = self.forward_input_update_nn(input_feature_vectors[input_idx])

            # Now, perform the forward update of the ReLU layer embedding vectors of each layer. Initialise the variable
            # which will be counting the number of ReLUs, thus matching the appropriate embedding vectors to ReLUs
            relu_layer_idx = 0
            for layer_idx, layer in enumerate(self.neural_network.children()):
                pass
                # TODO

    def compute_scores(self):
        """
        This function computes the scores for all the input nodes
        """
        # Initialise the row tensor of tensors of scores where each tensor corresponds to the scores associated with a
        # particular pixel (size hasn't been computed yet)
        scores = torch.tensor([0])

        # For each input node, pass the corresponding embedding vector through the Score Computation NN
        for input_idx in range(self.input_embeddings.size()[0]):
            pixel_scores = self.score_computation_nn(self.input_embeddings[input_idx])

            # During the first loop, resize the tensor containing scores to the correct size
            if input_idx == 0:
                scores = torch.zeros(torch.Size([self.input_embeddings.size()[0], pixel_scores.size()[0]]))

            scores[input_idx] = pixel_scores

        return scores


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


class ForwardReluUpdateNN(nn.Module):
    """
    This class represents the neural network which performs the forward update on the ReLU hidden layer nodes
    """
    def __init__(self, feature_vector_size, hidden_layer_size, embedding_vector_size):
        super(ForwardReluUpdateNN, self).__init__()

        # Initialise the layers for obtaining information from the local features
        self.linear_local_1 = nn.Linear(feature_vector_size, hidden_layer_size)
        self.linear_local_2 = nn.Linear(hidden_layer_size, hidden_layer_size)

        # Initialise the layers for obtaining information from the previous neighbour embedding vectors
        self.linear_neighbour_1 = nn.Linear(2 * embedding_vector_size, hidden_layer_size)
        self.linear_neighbour_2 = nn.Linear(hidden_layer_size, hidden_layer_size)

        # Finally, initialise the layers for combining information from the local features and the previous neighbour
        # embedding vectors
        self.linear_combine_1 = nn.Linear(2 * hidden_layer_size, hidden_layer_size)
        self.linear_combine_2 = nn.Linear(hidden_layer_size, embedding_vector_size)

    def forward(self, local_feature_vector, transformed_previous_neighbour_embeddings):
        # First, get information from the local feature vector. If the hidden layer node currently considered is
        # unambiguous, set the information vector to the zero vector. Otherwise, pass it through the corresponding
        # network layers
        # TODO
        local_features_info = self.linear_local_2(f.relu(self.linear_local_1(local_feature_vector)))

        # Second, get information from the transformed previous neighbour embedding vectors
        previous_neighbour_embeddings_info = self.linear_neighbour_2(f.relu(
            self.linear_neighbour_1(transformed_previous_neighbour_embeddings)))

        # Finally, combine the information from the local features and the transformed previous neighbour embedding
        # vectors
        combined_info = torch.cat([local_features_info, previous_neighbour_embeddings_info])
        return self.linear_combine_2(f.relu(self.linear_combine_1(combined_info)))


class ForwardOutputUpdateNN(nn.Module):
    """
    This class represents the neural network which performs the forward update on the output node
    """
    def __init__(self, feature_vector_size, hidden_layer_size, embedding_vector_size):
        super(ForwardOutputUpdateNN, self).__init__()

        # Initialise the layer for obtaining information from the local features
        self.linear_local = nn.Linear(feature_vector_size, hidden_layer_size)

        # Initialise the layers for combining information from the local features and the last hidden layer embeddings
        self.linear_combine_1 = nn.Linear(hidden_layer_size + embedding_vector_size, hidden_layer_size)
        self.linear_combine_2 = nn.Linear(hidden_layer_size, embedding_vector_size)

    def forward(self, local_feature_vector, transformed_last_hidden_layer_embeddings):
        # First, get information from the local feature vector
        local_features_info = f.relu(self.linear_local(local_feature_vector))

        # Combine the information from the local features and the transformed last hidden layer embeddings
        combined_info = torch.cat([local_features_info, transformed_last_hidden_layer_embeddings])
        return self.linear_combine_2(f.relu(self.linear_combine_1(combined_info)))


class BackwardReluUpdateNN(nn.Module):
    """
    This class represents the neural network which performs the backward update on the hidden layer nodes
    """
    def __init__(self, feature_vector_size, hidden_layer_size, embedding_vector_size):
        super(BackwardReluUpdateNN, self).__init__()

        # Initialise the layers for obtaining information from the local features
        self.linear_local_1_1 = nn.Linear(feature_vector_size, hidden_layer_size)
        self.linear_local_1_2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear_local_1_3 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear_local_2_1 = nn.Linear(3 * hidden_layer_size, hidden_layer_size)
        self.linear_local_2_2 = nn.Linear(hidden_layer_size, hidden_layer_size)

        # Initialise the layers for obtaining information from the next neighbour embedding vectors
        self.linear_neighbour_1 = nn.Linear(2 * embedding_vector_size, hidden_layer_size)
        self.linear_neighbour_2 = nn.Linear(hidden_layer_size, hidden_layer_size)

        # Finally, initialise the layers for combining information from the local features and the next neighbour
        # embedding vectors
        self.linear_combine_1 = nn.Linear(2 * hidden_layer_size, hidden_layer_size)
        self.linear_combine_2 = nn.Linear(hidden_layer_size, embedding_vector_size)

    def forward(self, local_feature_vector, transformed_next_layer_embeddings):
        # First, implement the 1st stage of getting information about the local feature vector. If the hidden layer node
        # currently considered is unambiguous, set the output from the first stage to the zero vector. Otherwise, pass
        # it through the layers of the 1st stage network
        # TODO
        local_features_info_temp = self.linear_local_1_3(f.relu(self.linear_local_1_2(f.relu(
            self.linear_local_1_1(local_feature_vector)))))

        # Now implement the 2nd stage of getting information about the local feature vector. If the output from the
        # 1st stage is zero, then set the information vector to the zero vector. Otherwise, pass it through the layers
        # of the 2nd stage network
        # TODO
        local_features_info = self.linear_local_2_2(f.relu(self.linear_local_2_1(local_features_info_temp)))

        # Second, get information from the transformed next neighbour embedding vectors
        next_neighbour_embeddings_info = self.linear_neighbour_2(f.relu(
            self.linear_neighbour_1(transformed_next_layer_embeddings)))

        # Finally, combine the information from the local features and the transformed next neighbour embedding
        # vectors
        combined_info = torch.cat([local_features_info, next_neighbour_embeddings_info])
        return self.linear_combine_2(f.relu(self.linear_combine_1(combined_info)))


class BackwardInputUpdateNN(nn.Module):
    """
    This class represents the neural network which performs the backward update on the input nodes
    """
    def __init__(self, feature_vector_size, hidden_layer_size, embedding_vector_size):
        super(BackwardInputUpdateNN, self).__init__()

        # Initialise the layers for obtaining information from the local features
        self.linear_local_1 = nn.Linear(feature_vector_size, hidden_layer_size)
        self.linear_local_2 = nn.Linear(hidden_layer_size, hidden_layer_size)

        # Initialise the layers for combining information from the local features and the first hidden layer embeddings
        self.linear_combine_1 = nn.Linear(hidden_layer_size + embedding_vector_size, hidden_layer_size)
        self.linear_combine_2 = nn.Linear(hidden_layer_size, embedding_vector_size)

    def forward(self, local_feature_vector, transformed_last_hidden_layer_embeddings):
        # First, get information from the local feature vector
        local_features_info = self.linear_local_2(f.relu(self.linear_local_1(local_feature_vector)))

        # Combine the information from the local features and the transformed first hidden layer embeddings
        combined_info = torch.cat([local_features_info, transformed_last_hidden_layer_embeddings])
        return self.linear_combine_2(f.relu(self.linear_combine_1(combined_info)))


class ScoreComputationNN(nn.Module):
    """
    This class represents the neural network which computes the scores for all possible input domain update methods
    """
    def __init__(self, embedding_vector_size, hidden_layer_size, number_of_update_methods):
        super(ScoreComputationNN, self).__init__()

        # Assuming this network is a 2-layer fully-connected network, initialise the two required layers
        self.linear_1 = nn.Linear(embedding_vector_size, hidden_layer_size)
        self.linear_2 = nn.Linear(hidden_layer_size, number_of_update_methods)

    def forward(self, input_embedding_vector):
        return self.linear_2(f.relu(self.linear_1(input_embedding_vector)))
