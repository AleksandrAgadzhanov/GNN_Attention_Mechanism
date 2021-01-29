import torch
import torch.nn as nn
from plnn.modules import Flatten
from GNN_framework.helper_functions import transform_embedding_vectors, get_numbers_of_connecting_nodes
from GNN_framework.auxiliary_neural_networks import ForwardInputUpdateNN, ForwardReluUpdateNN, ForwardOutputUpdateNN, \
    BackwardReluUpdateNN, BackwardInputUpdateNN, BoundsUpdateNN
from torch.nn import functional as f


class GraphNeuralNetwork:
    """
    This class represents the overall Graph Neural Network and its functionality
    """

    # When the GNN is initialised, a graph is created which has the same structure as the input neural network but each
    # proper node (input, ReLU and output) now containing the corresponding embedding vector filled with zeros.
    # In addition, all the auxiliary neural networks are initialised including the ones which perform the forward and
    # backward update operations and the one which computes the new bounds of all input nodes
    def __init__(self, neural_network, input_size, input_feature_size, relu_feature_size, output_feature_size,
                 embedding_vector_size=64, auxiliary_hidden_size=64, training_mode=False):
        # Store the underlying neural network as a field of the GNN
        self.neural_network = neural_network

        # Initialise a row tensor of tensors each corresponding to a particular pixel
        self.input_embeddings = torch.zeros([embedding_vector_size, *input_size[1:]])

        # Initialise the list of embeddings of each ReLU hidden layer
        self.relu_embeddings_list = []

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
                self.relu_embeddings_list.append(torch.zeros([embedding_vector_size, *test_image.size()[1:]]))

        # Finally, initialise the output embedding vectors for the output nodes using the test output from the network
        self.output_embeddings = torch.zeros([embedding_vector_size, *test_image.size()[1:]])

        # Now, initialise all the auxiliary neural networks
        self.forward_input_update_nn = ForwardInputUpdateNN(input_feature_size, auxiliary_hidden_size,
                                                            embedding_vector_size, training_mode)
        self.forward_relu_update_nn = ForwardReluUpdateNN(relu_feature_size, auxiliary_hidden_size,
                                                          embedding_vector_size, training_mode)
        self.forward_output_update_nn = ForwardOutputUpdateNN(output_feature_size, auxiliary_hidden_size,
                                                              embedding_vector_size, training_mode)
        self.backward_relu_update_nn = BackwardReluUpdateNN(relu_feature_size, auxiliary_hidden_size,
                                                            embedding_vector_size, training_mode)
        self.backward_input_update_nn = BackwardInputUpdateNN(input_feature_size, auxiliary_hidden_size,
                                                              embedding_vector_size, training_mode)
        self.bounds_update_nn = BoundsUpdateNN(embedding_vector_size, auxiliary_hidden_size, training_mode)

        # Finally, store the training mode flag
        self.training_mode = training_mode

    def reset_input_embedding_vectors(self):
        """
        This function resets all the input embedding vectors to the zero vectors
        """
        self.input_embeddings = torch.zeros(self.input_embeddings.size())

    def update_embedding_vectors(self, input_feature_vectors, relu_feature_vectors_list, output_feature_vectors,
                                 num_updates=2):
        """
        This function performs a series of forward and backward updates on all the embedding vectors until convergence
        is reached (specified by the number of updates input variable).
        """
        # Perform one complete forward and backward update a number of times specified
        for i in range(num_updates):

            # First, perform the forward update of the input embedding vectors if they are still all zero (only happens
            # during the first update)
            if torch.eq(self.input_embeddings, torch.zeros(self.input_embeddings.size())).all().item():
                self.forward_update_input_embeddings(input_feature_vectors)

            # Now, perform the forward update of the ReLU layers and output layer embedding vectors using a dedicated
            # function
            size_before_flattening = self.forward_update_relu_output_embeddings(relu_feature_vectors_list,
                                                                                output_feature_vectors)

            self.backward_update_relu_input_embeddings(input_feature_vectors, relu_feature_vectors_list,
                                                       size_before_flattening)

    def forward_update_input_embeddings(self, input_feature_vectors):
        """
        This function performs the forward update on the input layer embedding vectors.
        """
        # Reshape the input embedding vectors for the time of update to be the tensor which has the shape
        # [embedding_size, image_dimensions_product], storing the original size to reshape them back after the update
        original_size = self.input_embeddings.size()
        self.input_embeddings = self.input_embeddings.reshape(self.input_embeddings.size()[0], -1)

        # Perform the forward update on all the embedding vectors at once in order to avoid inplace operations
        self.input_embeddings = torch.transpose(self.forward_input_update_nn(
            torch.transpose(input_feature_vectors, 1, 0)), 1, 0)

        # Reshape the input embedding vectors to have the same size as before the update, since it will be easier
        self.input_embeddings = self.input_embeddings.reshape(original_size)

    def forward_update_relu_output_embeddings(self, relu_feature_vectors_list, output_feature_vectors):
        """
        This function performs the forward updates of all the ReLU layers and output layer embedding vectors based on
        the techniques outlined in the "NN Branching for NN Verification" paper.
        """
        # Initialise the variable which will be counting the number of ReLUs, thus matching the appropriate features and
        # embedding vectors to corresponding ReLUs, and the variable which will keep track of the output of each layer
        relu_layer_idx = 0
        embedding_vectors = self.input_embeddings.clone()

        # Initialise an extra variable which will be needed during the backward update - the size before flattening
        size_before_flattening = torch.Size([0])

        # Now go over each layer in turn, taking care to distinguish between the different types of layers
        for layer in self.neural_network.children():

            # If the layer is linear or convolutional, pass the embedding vectors through it, without applying the bias
            if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
                layer_bias = layer.bias.data
                layer.bias.data = torch.zeros(layer.bias.data.size())
                embedding_vectors = layer(embedding_vectors)
                layer.bias.data = layer_bias

            # If the layer is of type Flatten, simply pass the embedding vectors through it, storing the shape before
            # flattening in an extra variable
            elif type(layer) == Flatten:
                size_before_flattening = torch.Size(embedding_vectors.size())
                embedding_vectors = layer(embedding_vectors)

            # If the layer is a ReLU layer, then update the embedding vectors accordingly
            elif type(layer) == nn.ReLU:
                # First, extract the appropriate feature vectors tensor from the list
                relu_feature_vectors = relu_feature_vectors_list[relu_layer_idx]

                # Reshape the embedding vectors to be of size [embedding_size, product_of_other_dimensions] during the
                # update, storing its original size to reshape it back after the update
                original_size = embedding_vectors.size()
                embedding_vectors = embedding_vectors.reshape(embedding_vectors.size()[0], -1)

                # Transform all the embedding vectors so that they can enter the appropriate neural network
                transformed_embedding_vectors = transform_embedding_vectors(embedding_vectors, relu_feature_vectors)

                # Update the embedding vectors all at once in order to avoid inplace operations
                embedding_vectors = torch.transpose(self.forward_relu_update_nn(torch.transpose(
                    relu_feature_vectors, 1, 0), torch.transpose(transformed_embedding_vectors, 1, 0)), 1, 0)

                # Reshape the embedding vectors tensor to have the original size
                embedding_vectors = embedding_vectors.reshape(original_size)

                # Set the corresponding element of the storage to the updated embedding vectors tensor and increment the
                # ReLU layer counter
                self.relu_embeddings_list[relu_layer_idx] = embedding_vectors.clone()
                relu_layer_idx += 1

            # Otherwise, the layer type hasn't been considered yet
            else:
                raise NotImplementedError

        # At this point the output layer is reached, so it is left to update its embedding vectors based on the final
        # form of the propagated embeddings using the same procedure as for the ReLU layers
        original_size = embedding_vectors.size()
        embedding_vectors = embedding_vectors.reshape(embedding_vectors.size()[0], -1)
        embedding_vectors = torch.transpose(self.forward_output_update_nn(
            torch.transpose(output_feature_vectors, 1, 0), torch.transpose(embedding_vectors, 1, 0)), 1, 0)
        embedding_vectors = embedding_vectors.reshape(original_size)
        self.output_embeddings = embedding_vectors.clone()

        return size_before_flattening

    def backward_update_relu_input_embeddings(self, input_feature_vectors, relu_feature_vectors_list,
                                              size_before_flattening):
        """
        This function performs the backward updates of all the ReLU layers embedding vectors based on the techniques
        outlined in the "NN Branching for NN Verification" paper.
        """
        # Initialise the variable which will be keeping track of ReLUs, thus matching the appropriate features and
        # embedding vectors to corresponding ReLUs, and the variable which will keep track of the output of each layer
        # (now initialised at the output since this is a backward update)
        relu_layer_idx = len(self.relu_embeddings_list) - 1
        embedding_vectors = self.output_embeddings.clone()

        # Now go backward over each layer in turn, taking care to distinguish between the different types of layers
        for layer_idx, layer in enumerate(reversed(list(self.neural_network.children()))):

            # If the layer is linear, construct the layer with the same weights and zero biases, but passing in the
            # opposite direction by swapping the inputs and outputs. Also set all the required_grad parameters of this
            # layer to False to avoid creating unnecessary gradient computation graphs if not in training mode
            if type(layer) == nn.Linear:
                backwards_linear_layer = nn.Linear(layer.out_features, layer.in_features)
                backwards_linear_layer.weight.data = torch.transpose(layer.weight.data, 1, 0)
                backwards_linear_layer.bias.data = torch.zeros(backwards_linear_layer.bias.data.size())
                if not self.training_mode:
                    for parameter in backwards_linear_layer.parameters():
                        parameter.requires_grad = False
                embedding_vectors = backwards_linear_layer(embedding_vectors)

            # If the layer is convolutional, construct the layer in the similar way to above (effectively performing
            # deconvolution), but this time specifying the full set of parameters. Also set requires_grad to False
            elif type(layer) == nn.Conv2d:
                backwards_conv_layer = nn.ConvTranspose2d(layer.out_channels, layer.in_channels,
                                                          kernel_size=layer.kernel_size, stride=layer.stride,
                                                          padding=layer.padding, dilation=layer.dilation,
                                                          groups=layer.groups)
                backwards_conv_layer.weight.data = layer.weight.data
                if not self.training_mode:
                    for parameter in backwards_conv_layer.parameters():
                        parameter.requires_grad = False
                backwards_conv_layer.bias.data = torch.zeros(backwards_conv_layer.bias.data.size())

                # If the next layer exists and is a ReLU layer or if this is the last layer (before the input one),
                # find the number of connecting nodes in the convolutional layer for each ReLU layer node by calling the
                # appropriate function, then divide the embedding vectors tensor by them
                if (layer_idx <= len((list(self.neural_network.children()))) - 2 and
                    type(list(reversed(list(self.neural_network.children())))[layer_idx + 1]) == nn.ReLU) or \
                        layer_idx == len((list(self.neural_network.children()))) - 1:
                    numbers_of_connecting_nodes = get_numbers_of_connecting_nodes(backwards_conv_layer,
                                                                                  embedding_vectors.size(),
                                                                                  self.training_mode)
                    embedding_vectors = backwards_conv_layer(embedding_vectors)
                    embedding_vectors = embedding_vectors / numbers_of_connecting_nodes
                # Otherwise, simply pass the embedding vectors through without scaling
                else:
                    embedding_vectors = backwards_conv_layer(embedding_vectors)

            # If the layer is of type Flatten, then expansion to the size present before the tensor was flattened during
            # the forward update should be performed instead of flattening
            elif type(layer) == Flatten:
                embedding_vectors = embedding_vectors.reshape(size_before_flattening)

            # If the layer is a ReLU layer, then update the embedding vectors accordingly
            elif type(layer) == nn.ReLU:
                # First, extract the appropriate feature vectors tensor from the list
                relu_feature_vectors = relu_feature_vectors_list[relu_layer_idx]

                # Reshape the embedding vectors to be of size [embedding_size, product_of_other_dimensions] during the
                # update, storing its original size to reshape it back after the update
                original_size = embedding_vectors.size()
                embedding_vectors = embedding_vectors.reshape(embedding_vectors.size()[0], -1)

                # Transform all the embedding vectors so that they can enter the appropriate neural network
                transformed_embedding_vectors = transform_embedding_vectors(embedding_vectors, relu_feature_vectors)

                # Update all embedding vectors at once to avoid inplace operations
                embedding_vectors = torch.transpose(self.backward_relu_update_nn(torch.transpose(
                    relu_feature_vectors, 1, 0), torch.transpose(transformed_embedding_vectors, 1, 0)), 1, 0)

                # Reshape the embedding vectors tensor to have the original size
                embedding_vectors = embedding_vectors.reshape(original_size)

                # Set the corresponding element of the storage to the updated embedding vectors tensor and decrement the
                # ReLU layer counter
                self.relu_embeddings_list[relu_layer_idx] = embedding_vectors.clone()
                relu_layer_idx -= 1

            # Otherwise, the layer type hasn't been considered yet
            else:
                raise NotImplementedError

        # At this point the input layer is reached, so it is left to update its embedding vectors based on the final
        # form of the propagated embeddings using the same procedure as for the ReLU layers
        original_size = embedding_vectors.size()
        embedding_vectors = embedding_vectors.reshape(embedding_vectors.size()[0], -1)
        embedding_vectors = torch.transpose(self.backward_input_update_nn(
            torch.transpose(input_feature_vectors, 1, 0), torch.transpose(embedding_vectors, 1, 0)), 1, 0)
        embedding_vectors = embedding_vectors.reshape(original_size)
        self.input_embeddings = embedding_vectors.clone()

    def compute_updated_bounds(self, old_lower_bound, old_upper_bound):
        """
        This function computes the updated lower and upper bounds of the input domain by passing the input embedding
        vectors through the appropriate neural network and then constraining its output in a meaningful way.
        """
        # Reshape the input embeddings to the size which is easy to iterate over, storing the original size
        original_size = self.input_embeddings.size()
        self.input_embeddings = self.input_embeddings.reshape(self.input_embeddings.size()[0], -1)

        # Initialise the row tensor of tensors of new lower and upper bounds where each tensor corresponds to the new
        # lower and upper bound associated with a particular pixel
        new_lower_bound_and_offset = torch.zeros([2, self.input_embeddings.size()[-1]])

        # Pass all the embedding vectors through the Bounds Update NN at once to avoid inplace operations
        new_lower_bound_and_offset = torch.transpose(self.bounds_update_nn(
            torch.transpose(self.input_embeddings, 1, 0)), 1, 0)

        # Reshape the input embeddings to the original size
        self.input_embeddings = self.input_embeddings.reshape(original_size)

        # Initialise the new lower and the offset tensors
        new_lower_bound = new_lower_bound_and_offset[0, :].reshape(old_lower_bound.size())
        offset = new_lower_bound_and_offset[1, :].reshape(old_lower_bound.size())

        # Now the constraints should be applied on new lower bounds and offsets.
        # 1. If the obtained lower bound is smaller than the old lower bound, set the new lower bound to the old one; if
        # it is larger than the old upper bound, set it to the old upper bound
        new_lower_bound = torch.max(new_lower_bound, old_lower_bound)
        new_lower_bound = torch.min(new_lower_bound, old_upper_bound)

        # 2. If some elements of the offset are negative, set them to zero
        offset = f.relu(offset)

        # Initialise the upper bound tensor
        new_upper_bound = new_lower_bound + offset

        # 3. Finally, if the new upper bound is bigger than the old upper bound, set it to the old upper bound
        new_upper_bound = torch.min(new_upper_bound, new_lower_bound)
        if torch.eq(new_upper_bound, new_lower_bound).all().item():
            new_lower_bound = torch.add(new_upper_bound, new_lower_bound) / 2
        return new_lower_bound, new_upper_bound

    def parameters(self):
        """
        This function returns an iterator over all the parameters of the 6 auxiliary neural networks.
        """
        # First, put all the auxiliary neural networks in a list
        gnn_neural_networks = [self.forward_input_update_nn,
                               self.forward_relu_update_nn,
                               self.forward_output_update_nn,
                               self.backward_relu_update_nn,
                               self.backward_input_update_nn,
                               self.bounds_update_nn]

        # Now use the "yield" keyword to return the generator object on all the parameters of all the above networks
        for neural_network in gnn_neural_networks:
            for parameter in neural_network.parameters():
                yield parameter.requires_grad_(True)
