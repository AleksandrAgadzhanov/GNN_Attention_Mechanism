import torch
from exp_utils.model_utils import add_single_prop


def match_with_subset(subset, images, true_labels, test_labels, epsilons):
    """
    This function updates the lists of images, true labels, test labels and epsilons based on the given subset of
    indices
    """
    original_length = len(images)
    for i in range(original_length - 1, -1, -1):
        if i not in subset:
            images.pop(i)
            true_labels.pop(i)
            test_labels.pop(i)
            epsilons.pop(i)
    return images, true_labels, test_labels, epsilons


def simplify_model(model, true_label, test_label):
    """
    This function takes the model and the true and test labels for a particular property and simplifies the model by
    merging the last two linear layers into one so that the output of the network is a single node whose value is the
    difference between logits of the true and test classes
    """
    # First, get the list of all layers with the final one included. The output of the network is now the difference
    # between the logits of the true and test class
    all_layers = add_single_prop(list(model.children()), true_label, test_label)

    # Construct the simplified model from the list of layers
    simplified_model = torch.nn.Sequential(*all_layers)

    return simplified_model


def perturb_image(lower_bound, upper_bound):
    difference = torch.add(upper_bound, -lower_bound)
    perturbation = torch.mul(difference, torch.rand(difference.size()))
    perturbed_image = torch.add(lower_bound, perturbation)
    return perturbed_image


def gradient_ascent(simplified_model, perturbed_image, lower_bound, upper_bound, pgd_learning_rate, num_iterations,
                    return_loss=False):
    """
    This function performs Gradient Ascent on the specified property given the bounds on the input and, if it didn't
    lead to positive loss, outputs the information about gradients and the last version of the optimized variable.
    """
    # First of all, set the requires_grad parameter of the perturbed image to True to make it suitable for optimization
    perturbed_image.requires_grad = True

    # Initialise the relevant optimiser and the tensor to store the gradients to be later used for gradient information
    optimizer = torch.optim.Adam([perturbed_image], lr=pgd_learning_rate)
    gradients = torch.zeros([num_iterations, *perturbed_image.size()])
    loss = float('inf')

    # Perform Gradient Ascent for a specified number of epochs
    for iteration_idx in range(num_iterations):
        # The output of the network is the difference between the logits of the correct and the test class which is the
        # same as -loss, but since Gradient Ascent is performed, it is this difference that must be minimised
        loss = simplified_model(perturbed_image)

        # If the difference between the logit of the test class and the logit of the true class is positive, then the
        # PGD attack was successful and gradient ascent can be stopped
        if loss < 0:
            if return_loss:
                return loss
            else:
                return True, None, None

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store the current gradient in the list
        gradients[iteration_idx] = perturbed_image.grad.clone()

        # Clip the values of the perturbed image so that they are within the allowed perturbation magnitude. This
        # operation isn't related to optimisation, hence it is wrapped with torch.no_grad()
        with torch.no_grad():
            perturbed_image[:] = torch.max(torch.min(perturbed_image, upper_bound), lower_bound)

    # If the flag has not been set yet but the perturbation resulted in the model predicting the test class instead of
    # the true one during the last iteration, return the True successful attack flag
    if simplified_model(perturbed_image) < 0:
        if return_loss:
            return loss
        else:
            return True, None, None

    # If the Gradient Ascent didn't lead to the changed prediction, then generate the dictionary containing information
    # about gradients and output it as well
    gradient_info_dict = generate_gradient_info_dict(gradients)

    # Set the requires_grad parameter of the perturbed image to False so that gradients with respect to it aren't
    # computed outside of this function
    perturbed_image.requires_grad = False

    if return_loss:
        return loss
    else:
        return False, perturbed_image, gradient_info_dict


def generate_gradient_info_dict(gradients):
    """
    This function generates the dictionary of containing information about gradients during the PGD attack.
    """
    # First take the absolute value since it is the magnitudes of gradients that matter since loss is always moving in
    # the same direction (increasing) and the pixel values may decrease or increase
    gradient_magnitudes = torch.abs(gradients)
    gradient_magnitudes = gradient_magnitudes.reshape(gradient_magnitudes.size()[0], -1)

    # Initialise the output dictionary and add the row last gradient magnitude to the dictionary straight away
    gradient_info_dict = {'last gradient': gradient_magnitudes[-1]}

    # Transform the list of consequent gradients into the list of pixel gradients by transposing the above tensor
    gradient_magnitudes_pixels = torch.transpose(gradient_magnitudes, 1, 0)

    # Compute pixel-wise mean, median, maximum and minimum gradients as well as the standard deviation and store them
    mean_gradient_magnitudes_pixels = torch.mean(gradient_magnitudes_pixels, dim=1)
    median_gradient_magnitudes_pixels, _ = torch.median(gradient_magnitudes_pixels, dim=1)
    max_gradient_magnitudes_pixels, _ = torch.max(gradient_magnitudes_pixels, dim=1)
    min_gradient_magnitudes_pixels, _ = torch.min(gradient_magnitudes_pixels, dim=1)
    std_gradient_magnitudes_pixels = torch.std(gradient_magnitudes_pixels, dim=1)

    # Add the above statistics to the gradient information dictionary
    gradient_info_dict['mean gradient'] = mean_gradient_magnitudes_pixels
    gradient_info_dict['median gradient'] = median_gradient_magnitudes_pixels
    gradient_info_dict['maximum gradient'] = max_gradient_magnitudes_pixels
    gradient_info_dict['minimum gradient'] = min_gradient_magnitudes_pixels
    gradient_info_dict['gradient std'] = std_gradient_magnitudes_pixels

    return gradient_info_dict


def transform_embedding_vectors(embedding_vectors, local_feature_vectors):
    """
    This function transforms the embedding vectors which were propagated to the ReLU according to the technique
    outlined in the "NN Branching for NN Verification" paper.
    """
    # First, extract the lower and upper bounds from the local_feature_vectors tensor (located at rows 0 and 1
    # respectively of the local feature vectors matrix)
    lower_bounds = local_feature_vectors[0, :]
    upper_bounds = local_feature_vectors[1, :]

    # Compute the required ratios all at once in order to avoid inplace operations
    alphas = torch.where(lower_bounds > 0, torch.ones(lower_bounds.size()), torch.where(upper_bounds < 0, torch.zeros(
        lower_bounds.size()), torch.div(upper_bounds, torch.add(upper_bounds, -lower_bounds))))
    alphas_dashed = torch.where(lower_bounds > 0, torch.ones(alphas.size()), torch.where(upper_bounds < 0, torch.zeros(
        alphas.size()), torch.add(torch.ones(alphas.size()), -alphas)))

    # Finally, the transformed embedding vectors are defined in the following way
    product_1 = torch.mul(embedding_vectors, alphas)
    product_2 = torch.mul(embedding_vectors, alphas_dashed)
    transformed_embedding_vectors = torch.cat([product_1, product_2])

    return transformed_embedding_vectors


def get_numbers_of_connecting_nodes(backwards_conv_layer, input_size, training_mode=False):
    """
    This function computes the number of connecting nodes for each output node of a convolutional layer
    """
    # Initialise the test input of ones so that inputs are effectively counted for each output node in this way
    test_input = torch.ones([*[1 for _ in range(len(list(input_size[:-2])))], *input_size[-2:]])

    # Construct a copy of the passed convolutional layer but with 1 input and 1 output channel. Also set all weights of
    # the layer to ones and biases to zeros so that the numbers of connecting nodes appear in the output for each output
    # node. Finally, set all the required_grad parameters of its parameters to False if not in training mode
    modified_layer = torch.nn.ConvTranspose2d(1, 1, kernel_size=backwards_conv_layer.kernel_size,
                                              stride=backwards_conv_layer.stride, padding=backwards_conv_layer.padding,
                                              dilation=backwards_conv_layer.dilation,
                                              groups=backwards_conv_layer.groups)
    modified_layer.weight.data = torch.ones(modified_layer.weight.data.size())
    modified_layer.bias.data = torch.zeros(modified_layer.bias.data.size())

    if not training_mode:
        for parameter in modified_layer.parameters():
            parameter.requires_grad = False

    # Now pass the test input through the modified layer to obtain, for example, a tensor of size [1, 1, _, _]. It was
    # checked that when a tensor of larger size, e.g. [4, 16, _, _] is divided by it, each of the [_, _] matrices
    # is processed separately, which is exactly what is required
    numbers_of_connecting_nodes = modified_layer(test_input)

    return numbers_of_connecting_nodes
