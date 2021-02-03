import torch
import torch.nn as nn
from torch.nn import functional as f


class ForwardInputUpdateNN(nn.Module):
    """
    This class represents the neural network which performs the forward update on the input nodes
    """
    def __init__(self, feature_vector_size, hidden_layer_size, embedding_vector_size, training_mode):
        super(ForwardInputUpdateNN, self).__init__()
        self.linear_1 = nn.Linear(feature_vector_size, hidden_layer_size)
        self.linear_2 = nn.Linear(hidden_layer_size, embedding_vector_size)

        # If the training mode is off, then set all the required_grad parameters of all the layer parameters to False
        if not training_mode:
            for parameter in self.linear_1.parameters():
                parameter.requires_grad = False
            for parameter in self.linear_2.parameters():
                parameter.requires_grad = False

    def forward(self, input_feature_vectors):
        output = self.linear_1(input_feature_vectors)
        output = f.relu(output)
        output = self.linear_2(output)
        return output


class ForwardReluUpdateNN(nn.Module):
    """
    This class represents the neural network which performs the forward update on the ReLU hidden layer nodes
    """
    def __init__(self, feature_vector_size, hidden_layer_size, embedding_vector_size, training_mode):
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

        if not training_mode:
            for parameter in self.linear_local_1.parameters():
                parameter.requires_grad = False
            for parameter in self.linear_local_2.parameters():
                parameter.requires_grad = False
            for parameter in self.linear_neighbour_1.parameters():
                parameter.requires_grad = False
            for parameter in self.linear_neighbour_2.parameters():
                parameter.requires_grad = False
            for parameter in self.linear_combine_1.parameters():
                parameter.requires_grad = False
            for parameter in self.linear_combine_2.parameters():
                parameter.requires_grad = False

    def forward(self, local_feature_vectors, transformed_embedding_vectors):
        # First, get information from the local feature vectors. Then, if the hidden layer node is unambiguous (its
        # relaxation triangle intercept which is the last feature is zero), set its information vector to the zero
        # vector. Otherwise, pass it through the corresponding network layers. Do this for all the local feature vectors
        # at once to avoid the use of inplace operations
        local_features_info = self.linear_local_1(local_feature_vectors)
        local_features_info = f.relu(local_features_info)
        local_features_info = self.linear_local_2(local_features_info)

        local_features_info = torch.transpose(torch.where(local_feature_vectors[:, -1] == 0,
                                                          torch.transpose(torch.zeros([local_feature_vectors.size()[0],
                                                                                       self.linear_local_2.out_features]
                                                                                      ), 1, 0),
                                                          torch.transpose(local_features_info, 1, 0)), 1, 0)

        # Second, get information from the transformed previous neighbour embedding vectors
        previous_neighbour_embeddings_info = self.linear_neighbour_1(transformed_embedding_vectors)
        previous_neighbour_embeddings_info = f.relu(previous_neighbour_embeddings_info)
        previous_neighbour_embeddings_info = self.linear_neighbour_2(previous_neighbour_embeddings_info)

        # Finally, combine the information from the local features and the transformed previous neighbour embedding
        # vectors
        combined_info = torch.transpose(torch.cat([torch.transpose(local_features_info, 1, 0),
                                                   torch.transpose(previous_neighbour_embeddings_info, 1, 0)]), 1, 0)
        output = self.linear_combine_1(combined_info)
        output = f.relu(output)
        output = self.linear_combine_2(output)
        return output


class ForwardOutputUpdateNN(nn.Module):
    """
    This class represents the neural network which performs the forward update on the output node
    """
    def __init__(self, feature_vector_size, hidden_layer_size, embedding_vector_size, training_mode):
        super(ForwardOutputUpdateNN, self).__init__()

        # Initialise the layer for obtaining information from the local features
        self.linear_local = nn.Linear(feature_vector_size, hidden_layer_size)

        # Initialise the layers for combining information from the local features and the last hidden layer embeddings
        self.linear_combine_1 = nn.Linear(hidden_layer_size + embedding_vector_size, hidden_layer_size)
        self.linear_combine_2 = nn.Linear(hidden_layer_size, embedding_vector_size)

        if not training_mode:
            for parameter in self.linear_local.parameters():
                parameter.requires_grad = False
            for parameter in self.linear_combine_1.parameters():
                parameter.requires_grad = False
            for parameter in self.linear_combine_2.parameters():
                parameter.requires_grad = False

    def forward(self, local_feature_vectors, propagated_last_hidden_layer_embeddings):
        # First, get information from the local feature vector
        local_features_info = self.linear_local(local_feature_vectors)
        local_features_info = f.relu(local_features_info)

        # Combine the information from the local features and the transformed last hidden layer embeddings
        combined_info = torch.transpose(torch.cat([torch.transpose(
            local_features_info, 1, 0), torch.transpose(propagated_last_hidden_layer_embeddings, 1, 0)]), 1, 0)
        output = self.linear_combine_1(combined_info)
        output = f.relu(output)
        output = self.linear_combine_2(output)
        return output


class BackwardReluUpdateNN(nn.Module):
    """
    This class represents the neural network which performs the backward update on the hidden layer nodes
    """
    def __init__(self, feature_vector_size, hidden_layer_size, embedding_vector_size, training_mode):
        super(BackwardReluUpdateNN, self).__init__()

        # Initialise the layers for obtaining information from the local features
        self.linear_local_1 = nn.Linear(feature_vector_size, hidden_layer_size)
        self.linear_local_2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear_local_3 = nn.Linear(hidden_layer_size, hidden_layer_size)

        # Initialise the layers for obtaining information from the next neighbour embedding vectors
        self.linear_neighbour_1 = nn.Linear(2 * embedding_vector_size, hidden_layer_size)
        self.linear_neighbour_2 = nn.Linear(hidden_layer_size, hidden_layer_size)

        # Finally, initialise the layers for combining information from the local features and the next neighbour
        # embedding vectors
        self.linear_combine_1 = nn.Linear(2 * hidden_layer_size, hidden_layer_size)
        self.linear_combine_2 = nn.Linear(hidden_layer_size, embedding_vector_size)

        if not training_mode:
            for parameter in self.linear_local_1.parameters():
                parameter.requires_grad = False
            for parameter in self.linear_local_2.parameters():
                parameter.requires_grad = False
            for parameter in self.linear_local_3.parameters():
                parameter.requires_grad = False
            for parameter in self.linear_neighbour_1.parameters():
                parameter.requires_grad = False
            for parameter in self.linear_neighbour_2.parameters():
                parameter.requires_grad = False
            for parameter in self.linear_combine_1.parameters():
                parameter.requires_grad = False
            for parameter in self.linear_combine_2.parameters():
                parameter.requires_grad = False

    def forward(self, local_feature_vectors, transformed_next_layer_embeddings):
        # First, get information from the local feature vectors. Then, if the hidden layer node is unambiguous (its
        # relaxation triangle intercept which is the last feature is zero), set its information vector to the zero
        # vector. Otherwise, pass it through the corresponding network layers. Do this for all the local feature vectors
        # at once to avoid the use of inplace operations
        local_features_info = self.linear_local_1(local_feature_vectors)
        local_features_info = f.relu(local_features_info)
        local_features_info = self.linear_local_2(local_features_info)
        local_features_info = f.relu(local_features_info)
        local_features_info = self.linear_local_3(local_features_info)

        local_features_info = torch.transpose(torch.where(local_feature_vectors[:, -1] == 0,
                                                          torch.transpose(torch.zeros([local_feature_vectors.size()[0],
                                                                                       self.linear_local_3.out_features]
                                                                                      ), 1, 0),
                                                          torch.transpose(local_features_info, 1, 0)), 1, 0)

        # Second, get information from the transformed next neighbour embedding vectors
        next_neighbour_embeddings_info = self.linear_neighbour_1(transformed_next_layer_embeddings)
        next_neighbour_embeddings_info = f.relu(next_neighbour_embeddings_info)
        next_neighbour_embeddings_info = self.linear_neighbour_2(next_neighbour_embeddings_info)

        # Finally, combine the information from the local features and the transformed next neighbour embedding
        # vectors
        combined_info = torch.transpose(torch.cat([torch.transpose(local_features_info, 1, 0),
                                                   torch.transpose(next_neighbour_embeddings_info, 1, 0)]), 1, 0)
        output = self.linear_combine_1(combined_info)
        output = f.relu(output)
        output = self.linear_combine_2(output)
        return output


class BackwardInputUpdateNN(nn.Module):
    """
    This class represents the neural network which performs the backward update on the input nodes
    """
    def __init__(self, feature_vector_size, hidden_layer_size, embedding_vector_size, training_mode):
        super(BackwardInputUpdateNN, self).__init__()

        # Initialise the layers for obtaining information from the local features
        self.linear_local_1 = nn.Linear(feature_vector_size, hidden_layer_size)
        self.linear_local_2 = nn.Linear(hidden_layer_size, hidden_layer_size)

        # Initialise the layers for combining information from the local features and the first hidden layer embeddings
        self.linear_combine_1 = nn.Linear(hidden_layer_size + embedding_vector_size, hidden_layer_size)
        self.linear_combine_2 = nn.Linear(hidden_layer_size, embedding_vector_size)

        if not training_mode:
            for parameter in self.linear_local_1.parameters():
                parameter.requires_grad = False
            for parameter in self.linear_local_2.parameters():
                parameter.requires_grad = False
            for parameter in self.linear_combine_1.parameters():
                parameter.requires_grad = False
            for parameter in self.linear_combine_2.parameters():
                parameter.requires_grad = False

    def forward(self, local_feature_vectors, propagated_first_hidden_layer_embeddings):
        # First, get information from the local feature vector
        local_features_info = self.linear_local_1(local_feature_vectors)
        local_features_info = f.relu(local_features_info)
        local_features_info = self.linear_local_2(local_features_info)

        # Combine the information from the local features and the transformed first hidden layer embeddings
        combined_info = torch.transpose(torch.cat([torch.transpose(
            local_features_info, 1, 0), torch.transpose(propagated_first_hidden_layer_embeddings, 1, 0)]), 1, 0)
        output = self.linear_combine_1(combined_info)
        output = f.relu(output)
        output = self.linear_combine_2(output)
        return output


class BoundsUpdateNN(nn.Module):
    """
    This class represents the neural network which takes the input embedding vector and outputs a 2-dimensional tensor.
    Its first elements is the new updated lower bound and its second element is the offset from the new lower bound.
    """
    def __init__(self, embedding_vector_size, hidden_layer_size, training_mode):
        super(BoundsUpdateNN, self).__init__()

        # Assuming this network is a 2-layer fully-connected network, initialise the two required layers
        self.linear_1 = nn.Linear(embedding_vector_size, hidden_layer_size)
        self.linear_2 = nn.Linear(hidden_layer_size, 2)

        if not training_mode:
            for parameter in self.linear_1.parameters():
                parameter.requires_grad = False
            for parameter in self.linear_2.parameters():
                parameter.requires_grad = False

    def forward(self, input_embedding_vectors):
        output = self.linear_1(input_embedding_vectors)
        output = f.relu(output)
        output = self.linear_2(output)
        return output
