from exp_utils.model_utils import load_cifar_data, load_properties_data, add_single_prop
from plnn.conv_kwinter_kw import KWConvNetwork
from GNN_framework.GraphNeuralNetwork import GraphNeuralNetwork
import torch


def pgd_gnn_attack_properties(properties_filename, model_name, epsilon_factor, pgd_learning_rate, num_iterations,
                              num_epochs, num_updates, embedding_vector_size, auxiliary_hidden_size, num_update_methods,
                              subset=None):
    """
    This function acts aims to find adversarial examples for each property in the file specified. It acts as a container
    for the function which attacks each property in turn by calling this function for each property.
    """
    # Load all the required data for the images which were correctly verified by the model
    images, true_labels, image_indices, model = load_cifar_data(model_name)

    # Update the images and true_labels lists and load the lists of the test labels and epsilons such that they
    # correspond to the properties appearing in the properties dataframe which were correctly verified
    images, true_labels, test_labels, epsilons = load_properties_data(properties_filename, images, true_labels,
                                                                      image_indices)

    # If the subset of indices was specified for the purpose of reducing the time complexity, drop the elements of
    # images, true_labels, test_labels and epsilons not indicated in the subset indices
    if subset is not None:
        images, true_labels, test_labels, epsilons = match_with_subset(subset, images, true_labels, test_labels,
                                                                       epsilons)

    # Now attack each property in turn by calling the appropriate function
    num_properties_still_verified = 0  # counter of properties which are still verified after the PGD attack
    for i in range(len(images)):
        # First, simplify the network by adding the final layer and merging the last two layers into one, incorporating'
        # the information about the true and test classes into the network
        simplified_model = simplify_model(model, true_labels[i], test_labels[i])

        successful_attack_flag = pgd_gnn_attack_property(simplified_model, images[i], epsilons[i], epsilon_factor,
                                                         pgd_learning_rate, num_iterations, num_epochs, num_updates,
                                                         embedding_vector_size, auxiliary_hidden_size,
                                                         num_update_methods)

        # If the attack was unsuccessful, increase the counter
        if not successful_attack_flag:
            num_properties_still_verified += 1

    # Calculate the verification accuracy for the properties in the file provided after all the PGD attacks
    verification_accuracy = 100.0 * num_properties_still_verified / len(images)

    return verification_accuracy


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


def pgd_gnn_attack_property(simplified_model, image, epsilon, epsilon_factor, pgd_learning_rate, num_iterations,
                            num_epochs, num_updates, embedding_vector_size, auxiliary_hidden_size,
                            num_update_methods):
    """
    This function performs the PGD attack on the specified property characterised by its image, corresponding simplified
    model and epsilon value
    """
    # First, perturb the image randomly within the allowed bounds and perform a PGD attack
    lower_bound = torch.add(-epsilon * epsilon_factor, image)
    upper_bound = torch.add(epsilon * epsilon_factor, image)
    perturbed_image = perturb_image(lower_bound, upper_bound).requires_grad_(True)
    successful_attack_flag, gradient_info_dict = gradient_ascent(simplified_model, perturbed_image, lower_bound,
                                                                 upper_bound, pgd_learning_rate, num_iterations)

    # If the attack was successful, the procedure can be terminated and True can be returned
    if successful_attack_flag:
        return True

    # Otherwise, the GNN framework approach must be followed. First, generate the feature vectors for all layers
    input_feature_vectors = generate_input_feature_vectors(lower_bound, upper_bound, perturbed_image,
                                                           gradient_info_dict)
    relu_feature_vectors, output_feature_vectors = generate_relu_output_feature_vectors(simplified_model, lower_bound, upper_bound, image, perturbed_image, epsilon * epsilon_factor)

    # Initialise the GNN for the given network (which also initialises all the required auxiliary neural networks)
    gnn = GraphNeuralNetwork(simplified_model, image.size(), embedding_vector_size, input_feature_vectors[0].size()[0],
                             relu_feature_vectors[0][0].size()[0], output_feature_vectors[0].size()[0],
                             auxiliary_hidden_size, num_update_methods)

    # Follow the GNN framework approach for a specified number of epochs
    for i in range(num_epochs):
        # When the epoch is not the first one, reset the input embedding vectors since the forward input update function
        # only activates when the input embedding vectors are zero
        if i != 0:
            gnn.reset_input_embedding_vectors()

        # Perform a series of forward and backward updates of all the embedding vectors within the GNN
        gnn.update_embedding_vectors(input_feature_vectors, relu_feature_vectors, output_feature_vectors, num_updates)

        # Compute the scores for each image pixel
        pixel_scores = gnn.compute_scores()

        # Update the domain bounds for each pixel based on the pixel scores above
        lower_bound, upper_bound = update_domain_bounds(lower_bound, upper_bound, pixel_scores)

        # Perturb each pixel within the updated domain bounds
        perturbed_image = perturb_image(lower_bound, upper_bound).requires_grad_(True)

        # Perform a PGD attack given the new bounds and perturbation
        successful_attack_flag, gradient_info_dict = gradient_ascent(simplified_model, perturbed_image, lower_bound,
                                                                     upper_bound, pgd_learning_rate, num_iterations)

        # If the attack was successful, the procedure can be terminated and True can be returned, otherwise continue
        if successful_attack_flag:
            return True

    # If the limit on the number of epochs was reached and no PGD attack was successful, return False
    return False


def perturb_image(lower_bound, upper_bound):
    difference = torch.add(upper_bound, -lower_bound)
    perturbation = torch.mul(difference, torch.rand(difference.size()))
    perturbed_image = torch.add(lower_bound, perturbation)
    return perturbed_image


def gradient_ascent(simplified_model, perturbed_image, lower_bound, upper_bound, pgd_learning_rate, num_iterations):
    """
    This function performs Gradient Ascent on the specified property given the bounds on the input and, if it didn't
    lead to positive loss, outputs the information about gradients
    """
    # Initialise the relevant optimiser and the list to store the gradients to be later used for gradient information
    optimizer = torch.optim.Adam([perturbed_image], lr=pgd_learning_rate)
    gradients = []

    # Perform Gradient Ascent for a specified number of epochs
    for iteration in range(num_iterations):
        # The output of the network is the difference between the logits of the correct and the test class which is
        # the same as -loss, but since Gradient Ascent is performed, this difference must be minimised
        loss = simplified_model(perturbed_image)

        # If the difference between the logit of the test class and the logit of the true class is positive,
        # then the PGD attack was successful and gradient ascent can be stopped
        if loss < 0:
            return True, None

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store the current gradient in the list
        gradients.append(perturbed_image.grad.clone())

        # Clip the values of the perturbed image so that they are within the allowed perturbation magnitude
        # This operation isn't related to optimisation, hence it is wrapped with torch.no_grad()
        with torch.no_grad():
            perturbed_image[:] = torch.max(torch.min(perturbed_image, upper_bound), lower_bound)

    # If the flag has not been set yet but the perturbation resulted in the model predicting the test class instead of
    # the true one during the last iteration, return the True successful attack flag
    if simplified_model(perturbed_image) < 0:
        return True, None

    # If the Gradient Ascent didn't lead to the changed prediction, then generate the dictionary containing information
    # about gradients and output it as well
    gradient_info_dict = generate_gradient_info_dict(gradients)

    return False, gradient_info_dict


def generate_gradient_info_dict(gradients):
    """
    This function generates the dictionary of containing information about gradients during the PGD attack
    """
    # First take the absolute value since it is the magnitudes of gradients that matter since loss is always moving in
    # the same direction (increasing) and the pixel values may decrease or increase
    gradient_magnitudes = [torch.abs(gradient) for gradient in gradients]

    # Initialise the output dictionary and add the last gradient magnitude to the dictionary straight away
    gradient_info_dict = {'last gradient': gradient_magnitudes[-1]}

    # Transform the list of consequent gradients into the list of pixel gradients
    gradient_magnitudes_pixels = grouped_by_iteration_to_grouped_by_pixel(gradient_magnitudes)

    # Compute pixel-wise mean, median, maximum and minimum gradients as well as the standard deviation and store them
    mean_gradient_magnitudes_pixels = []
    median_gradient_magnitudes_pixels = []
    max_gradient_magnitudes_pixels = []
    min_gradient_magnitudes_pixels = []
    std_gradient_magnitudes_pixels = []

    for i in range(len(gradient_magnitudes_pixels)):
        mean_gradient_magnitudes_pixels.append(torch.mean(gradient_magnitudes_pixels[i]))
        median_gradient_magnitudes_pixels.append(torch.median(gradient_magnitudes_pixels[i]))
        max_gradient_magnitudes_pixels.append(torch.max(gradient_magnitudes_pixels[i]))
        min_gradient_magnitudes_pixels.append(torch.min(gradient_magnitudes_pixels[i]))
        std_gradient_magnitudes_pixels.append(torch.std(gradient_magnitudes_pixels[i]))

    # Reshape the lists of pixel statistics so that image shape is preserved
    mean_gradient_magnitude = torch.reshape(torch.tensor(mean_gradient_magnitudes_pixels), gradients[0].size())
    median_gradient_magnitude = torch.reshape(torch.tensor(median_gradient_magnitudes_pixels), gradients[0].size())
    max_gradient_magnitude = torch.reshape(torch.tensor(max_gradient_magnitudes_pixels), gradients[0].size())
    min_gradient_magnitude = torch.reshape(torch.tensor(min_gradient_magnitudes_pixels), gradients[0].size())
    std_gradient_magnitude = torch.reshape(torch.tensor(std_gradient_magnitudes_pixels), gradients[0].size())

    # Add the above statistics to the gradient information dictionary
    gradient_info_dict['mean gradient'] = mean_gradient_magnitude
    gradient_info_dict['median gradient'] = median_gradient_magnitude
    gradient_info_dict['maximum gradient'] = max_gradient_magnitude
    gradient_info_dict['minimum gradient'] = min_gradient_magnitude
    gradient_info_dict['gradient std'] = std_gradient_magnitude

    return gradient_info_dict


def grouped_by_iteration_to_grouped_by_pixel(gradients):
    """
    This function transforms the list of gradient magnitudes where each list element is the tensor represents the
    gradient at a particular epoch to the list where each element is the tensor representing the gradient of a
    particular pixel
    """
    # Reshape each gradient so that it becomes a row vector
    row_gradients = [gradient.view(-1) for gradient in gradients]

    # Since elements at the same indices now correspond to particular pixels, group them together
    row_gradients_grouped_by_pixel = []
    for pixel_idx in range(len(row_gradients[0])):
        pixel_gradients_list = []
        for gradient_idx in range(len(row_gradients)):
            pixel_gradients_list.append(row_gradients[gradient_idx][pixel_idx])
        row_gradients_grouped_by_pixel.append(torch.tensor(pixel_gradients_list))

    return row_gradients_grouped_by_pixel


def generate_input_feature_vectors(lower_bound, upper_bound, perturbed_image, gradient_info_dict):
    """
    This function generates the feature vectors for each input node from all the inputs provided
    """
    # Transform the gradient information dictionary into rows of gradient information tensors where each row has the
    # same size as the image and contains a piece of information about gradients for all pixels
    gradient_info_tensors = list(gradient_info_dict.values())
    gradient_info_row_tensors = [gradient_info_tensor.view(-1) for gradient_info_tensor in gradient_info_tensors]

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
            input_feature_vectors = torch.zeros(torch.Size([*lower_bound.size(),
                                                            len(pixel_info + pixel_gradient_info)]))
            input_feature_vectors = input_feature_vectors.reshape(-1, input_feature_vectors.size()[-1])

        # Finally, add the generated feature vector in the required place
        input_feature_vectors[input_idx] = torch.cat(pixel_info + pixel_gradient_info)

    return input_feature_vectors


# TODO
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
        perturbed_image = layer(perturbed_image)

        # If the current overall layer index is the same as the index before the ReLU layer, construct the tensor of
        # feature vectors
        if overall_layer_idx in pre_relu_indices:
            lower_bounds_before_relu = lower_bounds_all[overall_layer_idx].squeeze(0)
            relu_upper_bounds_before_relu = upper_bounds_all[overall_layer_idx].squeeze(0)
            node_values_before_relu = perturbed_image
            layer_bias_before_relu = layer.bias

            # If the ratio between the lower and upper bound is +ve, then the intercept of the relaxation triangle is
            # zero, otherwise it is easily obtained mathematically

    return None, None


def update_domain_bounds(old_lower_bound, old_upper_bound, scores):
    """
    This function updates the domain bounds based on the scores. For each input node, it chooses the update method
    corresponding to the maximum score among all scores for that node
    """
    # First, reshape the old lower and upper bounds to be row tensors
    old_lower_bound_row = old_lower_bound.view(-1)
    old_upper_bound_row = old_upper_bound.view(-1)

    # Initialise the new lower and upper bound tensors
    new_lower_bound_row = torch.zeros(old_lower_bound_row.size())
    new_upper_bound_row = torch.zeros(old_upper_bound_row.size())

    # The size of each score tensor represents the number of update methods chosen before. It is an odd number since one
    # update method is always for a node to retain the old bounds and all the other update methods (of which there is an
    # even number) symmetrically divide the original domain bounds with one division boundary being at the centre of the
    # old domain. Each score tensor  Make the decision for each input node
    for i in range(old_lower_bound_row.size()[0]):
        # Find the maximum score index in the corresponding score tensor
        max_score_index = torch.argmax(scores[i]).item()

        # If the maximum score index is the middle index, then retain the bounds as they are
        if max_score_index == 0:
            new_lower_bound_row[i] = old_lower_bound_row[i]
            new_upper_bound_row[i] = old_upper_bound_row[i]

        # Otherwise update the bounds according to the approach outlined above
        else:
            new_lower_bound_row[i] = old_lower_bound_row[i] + (max_score_index - 1) \
                                     * (old_upper_bound_row[i] - old_lower_bound_row[i]) / (scores[0].size()[0] - 1)
            new_upper_bound_row[i] = old_lower_bound_row[i] + max_score_index * (
                    old_upper_bound_row[i] - old_lower_bound_row[i]) / (scores[0].size()[0] - 1)

    # Finally, reshape the new lower and upper bounds so that they have the image shape
    new_lower_bound = new_lower_bound_row.reshape(old_lower_bound.size())
    new_upper_bound = new_upper_bound_row.reshape(old_upper_bound.size())

    return new_lower_bound, new_upper_bound


pgd_gnn_attack_properties('base_easy.pkl', 'cifar_base_kw', 1, 0.1, 10, 5, 5, 3, 10, 5, subset=[0])
