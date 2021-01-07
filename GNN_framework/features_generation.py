import torch
from plnn.conv_kwinter_kw import KWConvNetwork


def generate_input_feature_vectors(lower_bound, upper_bound, perturbed_image, gradient_info_dict):
    """
    This function generates the feature vectors for each input node from all the inputs provided.
    """
    # Transform the gradient information dictionary into rows of gradient information tensors where each row has the
    # same size as the image and contains a piece of information about gradients for all pixels
    gradient_info_row_tensors = list(gradient_info_dict.values())

    # Initialise the tensor variable to store the input feature vectors in (size hasn't been computed yet)
    input_feature_vectors = torch.tensor([0])

    # For each input node, generate the corresponding feature vector
    for input_idx in range(gradient_info_row_tensors[0].size()[0]):

        # Combine the information about the lower and upper bounds as well as the perturbed value of the pixel
        pixel_info = [torch.tensor([lower_bound.view(-1)[input_idx]]),
                      torch.tensor([upper_bound.view(-1)[input_idx]]),
                      torch.tensor([perturbed_image.view(-1)[input_idx]])]

        # Now extract all the information about gradients for a particular pixel
        pixel_gradient_info = []
        for i in range(len(gradient_info_row_tensors)):
            pixel_gradient_info.append(torch.tensor([gradient_info_row_tensors[i][input_idx]]))

        # During the first loop, resize the tensor containing input feature vectors to the correct size
        if input_idx == 0:
            input_feature_vectors = torch.zeros(torch.Size([len(pixel_info + pixel_gradient_info),
                                                            *lower_bound.size()]))
            input_feature_vectors = input_feature_vectors.reshape(input_feature_vectors.size()[0], -1)

        # Finally, add the generated feature vector in the required place
        input_feature_vectors[:, input_idx] = torch.cat(pixel_info + pixel_gradient_info)

    return input_feature_vectors


def generate_relu_output_feature_vectors(neural_network, input_lower_bound, input_upper_bound, image, perturbed_image,
                                         epsilon, image_is_bounded=False):
    """
    This function generates the feature vectors for each hidden layer and output node. Intermediate lower and upper
    bounds for each node are computed by applying the triangular approximation for each ReLU node and solving the
    resulting constrained optimisation problem
    """
    # First, create an instance of the KWConvNetwork class so that the function computing the bounds can be used
    kw_conv_network = KWConvNetwork(list(neural_network.children()))

    # Reshape the lower and upper bounds of the inputs for the function to be used properly
    input_lower_bound = input_lower_bound.squeeze(0)
    input_upper_bound = input_upper_bound.squeeze(0)
    input_domain = torch.zeros([3, 32, 32, 2])
    input_domain[:, :, :, 0] = input_lower_bound
    input_domain[:, :, :, 1] = input_upper_bound

    # Now call the build_the_model function which returns the lower and upper bounds for all layers (including the input
    # one) as lists of tensors which are shaped as, for example, [3, 32, 32] instead of the usual [1, 3, 32, 32], and
    # the list of indices of layers located just before the ReLU layers
    lower_bounds_all, upper_bounds_all, pre_relu_indices = kw_conv_network.build_the_model(input_domain, image,
                                                                                           epsilon,
                                                                                           image_is_bounded)

    # Initialise the list for storing the feature vectors of each ReLU layer and the tensor for storing the output
    # feature vectors (size hasn't been computed yet)
    relu_feature_vectors_list = []
    output_feature_vectors = torch.tensor([0])

    # Now iterate through the neural network layers and extract the information for the relevant feature vectors
    for layer_idx, layer in enumerate(list(neural_network.children())):
        overall_layer_idx = layer_idx + 1  # input layer is considered in pre_relu_indices, but not in neural_network
        perturbed_image = layer(perturbed_image)  # variable to keep track of layer outputs

        # If the current overall layer index is the same as the index before the ReLU layer, construct the tensor of
        # feature vectors and append it to the output list
        if overall_layer_idx in pre_relu_indices:
            # First extract all the relevant features and reshape them to be row tensors
            lower_bounds_before_relu = lower_bounds_all[overall_layer_idx].view(-1)
            upper_bounds_before_relu = upper_bounds_all[overall_layer_idx].view(-1)
            node_values_before_relu = perturbed_image.view(-1)

            # Layer bias has to be extracted with care since if the layer is convolutional, there is 1 bias value per
            # channel and has to be repeated to match the number of nodes
            layer_bias_before_relu = get_layer_bias_before_relu(layer, node_values_before_relu.size())

            # Relaxation triangle intercept is a condition dependent feature so will be computed separately using a
            # function defined below
            relaxation_triangle_intercepts = get_relaxation_triangle_intercepts(lower_bounds_before_relu,
                                                                                upper_bounds_before_relu)

            relu_feature_vectors_list.append(torch.stack([lower_bounds_before_relu,
                                                          upper_bounds_before_relu,
                                                          node_values_before_relu,
                                                          layer_bias_before_relu,
                                                          relaxation_triangle_intercepts]))

        # Finally, the last layer is the output one, so construct the corresponding feature vectors
        if layer_idx == len(list(neural_network.children())) - 1:
            output_lower_bounds = lower_bounds_all[-1].view(-1)
            output_upper_bounds = upper_bounds_all[-1].view(-1)
            output_node_values = perturbed_image.view(-1)
            output_bias = layer.bias.view(-1)

            output_feature_vectors = torch.stack([output_lower_bounds,
                                                  output_upper_bounds,
                                                  output_node_values,
                                                  output_bias])

    return relu_feature_vectors_list, output_feature_vectors


def get_layer_bias_before_relu(layer_before_relu, layer_before_relu_size):
    """
    This function generates the row tensor of biases present at the layer located just before the ReLU layer. It
    distinguishes between the case when this layer is convolutional, in which case there is 1 bias per channel which has
    to be repeated to match the number of nodes, and any other case.
    """
    # Initialise the output row tensor of appropriate size
    layer_bias_before_relu = torch.zeros(layer_before_relu_size).view(-1)

    # If the layer just before the ReLU layer is convolutional, follow the procedure described above
    if type(layer_before_relu) == torch.nn.Conv2d:
        constant_portion_size = int(layer_before_relu_size[0] / layer_before_relu.bias.size()[0])
        for bias_idx in range(layer_before_relu.bias.size()[0]):
            for num_repeat in range(constant_portion_size):
                layer_bias_before_relu[bias_idx * constant_portion_size + num_repeat] = layer_before_relu.bias[bias_idx]
    # Otherwise, simply set the output tensor
    else:
        layer_bias_before_relu = layer_before_relu.bias.view(-1)

    return layer_bias_before_relu


def get_relaxation_triangle_intercepts(lower_bounds_before_relu, upper_bounds_before_relu):
    """
    This function computes the intercepts of the relaxation triangles for all the ReLU layer nodes based on the lower
    and upper bounds of the layer just before the ReLU layer. The intercepts themselves act as features capturing the
    extent of convex relaxation that is introduced at each ReLU node.
    """
    # Initialise the output row tensor of appropriate size
    relaxation_triangle_intercepts = torch.zeros(lower_bounds_before_relu.size())

    # If the ratio between the lower and upper bound is +ve, then the intercept of the relaxation triangle is
    # zero, otherwise it is easily obtained as -ub * lb / (ub - lb) (ub and lb - upper and lower bounds)
    for i in range(lower_bounds_before_relu.size()[0]):
        if lower_bounds_before_relu[i] * upper_bounds_before_relu[i] > 0:
            relaxation_triangle_intercepts[i] = 0.0
        else:
            relaxation_triangle_intercepts[i] = - upper_bounds_before_relu[i] * lower_bounds_before_relu[i] / \
                                                (upper_bounds_before_relu[i] - lower_bounds_before_relu[i])

    return relaxation_triangle_intercepts
