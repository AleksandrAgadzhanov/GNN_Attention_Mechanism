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

    def forward(self, input_feature_vector):
        return self.linear_2(f.relu(self.linear_1(input_feature_vector)))


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

    def forward(self, local_feature_vector, transformed_embedding_vector):
        # First, get information from the local feature vector. If the hidden layer node currently considered is
        # unambiguous (its relaxation triangle intercept which is the last feature is zero), set the information vector
        # to the zero vector
        if local_feature_vector[-1].item() == 0.0:
            local_features_info = torch.zeros(self.linear_local_2.out_features)
        # Otherwise, pass it through the corresponding network layers
        else:
            local_features_info = self.linear_local_2(f.relu(self.linear_local_1(local_feature_vector)))

        # Second, get information from the transformed previous neighbour embedding vectors
        previous_neighbour_embeddings_info = self.linear_neighbour_2(f.relu(
            self.linear_neighbour_1(transformed_embedding_vector)))

        # Finally, combine the information from the local features and the transformed previous neighbour embedding
        # vectors
        combined_info = torch.cat([local_features_info, previous_neighbour_embeddings_info])
        return self.linear_combine_2(f.relu(self.linear_combine_1(combined_info)))


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

    def forward(self, local_feature_vector, propagated_last_hidden_layer_embeddings):
        # First, get information from the local feature vector
        local_features_info = f.relu(self.linear_local(local_feature_vector))

        # Combine the information from the local features and the transformed last hidden layer embeddings
        combined_info = torch.cat([local_features_info, propagated_last_hidden_layer_embeddings])
        return self.linear_combine_2(f.relu(self.linear_combine_1(combined_info)))


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

    def forward(self, local_feature_vector, transformed_next_layer_embeddings):
        # First, implement the 1st stage of getting information about the local feature vector. If the hidden layer node
        # currently considered is unambiguous (its relaxation triangle intercept which is the last feature is zero), set
        # the output from the first stage to zero the vector
        if local_feature_vector[-1].item() == 0.0:
            local_features_info = torch.zeros(self.linear_local_3.out_features)
        # Otherwise, pass it through the layers of the 1st stage network
        else:
            local_features_info = self.linear_local_3(f.relu(self.linear_local_2(f.relu(
                self.linear_local_1(local_feature_vector)))))

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

    def forward(self, local_feature_vector, propagated_first_hidden_layer_embeddings):
        # First, get information from the local feature vector
        local_features_info = self.linear_local_2(f.relu(self.linear_local_1(local_feature_vector)))

        # Combine the information from the local features and the transformed first hidden layer embeddings
        combined_info = torch.cat([local_features_info, propagated_first_hidden_layer_embeddings])
        return self.linear_combine_2(f.relu(self.linear_combine_1(combined_info)))


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

    def forward(self, input_embedding_vector):
        return self.linear_2(f.relu(self.linear_1(input_embedding_vector)))
