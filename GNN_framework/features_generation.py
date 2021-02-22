import torch
from plnn.run_attack import _run_lp_as_init


def generate_input_feature_vectors(lower_bound, upper_bound, perturbed_image, gradient_info_dict):
    """
    This function generates the feature vectors for each input node from all the inputs provided.
    """
    # Initialise the variable which will be storing all the gradient information tensors at once as its row tensors
    gradient_info_row_tensors = torch.zeros([len(gradient_info_dict), lower_bound.view(-1).size()[0]])

    # Fill in the values of the above variable
    for info_tensor_idx, info_tensor in enumerate(gradient_info_dict.values()):
        gradient_info_row_tensors[info_tensor_idx] = info_tensor

    # Combine the information about the lower and upper bounds as well as the perturbed value of the pixel
    pixel_info = torch.stack([lower_bound.view(-1), upper_bound.view(-1), perturbed_image.view(-1)])

    # Finally, construct the feature vectors variable by combining the pixel information with the gradient information
    input_feature_vectors = torch.cat([pixel_info, gradient_info_row_tensors])

    return input_feature_vectors


def generate_relu_output_feature_vectors(neural_network, input_lower_bound, input_upper_bound, perturbed_image,
                                         device='cpu'):
    """
    This function generates the feature vectors for each hidden layer and output node. Intermediate lower and upper
    bounds for each node are computed by applying the triangular approximation for each ReLU node and solving the
    resulting constrained optimisation problem.
    """
    # Prepare the inputs to the function computing the bounds
    verification_layers = list(neural_network.children())
    input_lower_bound = input_lower_bound.squeeze(0)
    input_upper_bound = input_upper_bound.squeeze(0)
    domain = torch.zeros([3, 32, 32, 2])
    domain[:, :, :, 0] = input_lower_bound
    domain[:, :, :, 1] = input_upper_bound

    # Make a call to the function which computes the bounds of all the network layers
    _, global_ub, intermediate_lbs, intermediate_ubs, _ = _run_lp_as_init(verification_layers, domain, device=device)

    # If CUDA was used in the above function, then send all of the outputs back to CPU
    if device == 'cuda' and torch.cuda.is_available():
        global_ub = global_ub.cpu()
        intermediate_lbs = [lb.cpu() for lb in intermediate_lbs]
        intermediate_ubs = [ub.cpu() for ub in intermediate_ubs]

    # Convert the function outputs into a form that is more convenient to work with. The first function output is a
    # tighter upper bound on the output node than the last element of the intermediate upper bounds list so use it as
    # the upper bound on the output. Intermediate bounds lists include the bounds on the inputs and output as first and
    # last elements respectively so store all the elements in between which contain the bounds on the ReLU layers.
    output_lower_bounds = intermediate_lbs[-1].view(-1)
    output_upper_bounds = global_ub.view(-1)
    relu_lower_bounds = intermediate_lbs[1:-1]
    relu_upper_bounds = intermediate_ubs[1:-1]

    # Initialise the list for storing the feature vectors of each ReLU layer and the tensor for storing the output
    # feature vectors (size hasn't been computed yet)
    relu_feature_vectors_list = []
    output_feature_vectors = torch.tensor([0])

    # Initialize the index of the intermediate bounds lists
    relu_layer_idx = 0

    # Now iterate through the neural network layers and extract the information for the relevant feature vectors
    for layer_idx, layer in enumerate(list(neural_network.children())):
        # Pass the variable which keeps track of the layer outputs through the current layer
        perturbed_image = layer(perturbed_image)

        # If the current layer is before the ReLU layer, extract all of the relevant features and store them in the list
        if layer_idx < len(list(neural_network.children())) - 1 and \
                type(list(neural_network.children())[layer_idx + 1]) == torch.nn.ReLU:
            # First extract all the relevant features and reshape them to be row tensors
            lower_bounds_before_relu = relu_lower_bounds[relu_layer_idx].view(-1)
            upper_bounds_before_relu = relu_upper_bounds[relu_layer_idx].view(-1)
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

            # Increment the index of the ReLU layer
            relu_layer_idx += 1

        # Finally, the last layer is the output one, so construct the corresponding feature vectors
        if layer_idx == len(list(neural_network.children())) - 1:
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
    # If the ratio between the lower and upper bound is +ve, then the intercept of the relaxation triangle is
    # zero, otherwise it is easily obtained as -ub * lb / (ub - lb) (ub and lb - upper and lower bounds)
    relaxation_triangle_intercepts = torch.where(lower_bounds_before_relu * upper_bounds_before_relu > 0,
                                                 torch.zeros(lower_bounds_before_relu.size()[0]),
                                                 torch.div(torch.mul(-upper_bounds_before_relu,
                                                                     lower_bounds_before_relu),
                                                           torch.add(upper_bounds_before_relu,
                                                                     -lower_bounds_before_relu)))

    return relaxation_triangle_intercepts
