from exp_utils.model_utils import load_cifar_data, load_properties_data, add_single_prop
from matplotlib import pyplot as plt
import torch


def pgd_gnn_attack_properties(properties_filename, model_name, epsilon_factor, pgd_learning_rate, num_epochs,
                              num_iterations, subset=None):
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
                                                         pgd_learning_rate, num_epochs, num_iterations)

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


def pgd_gnn_attack_property(simplified_model, image, epsilon, epsilon_factor, pgd_learning_rate, num_epochs,
                            num_iterations):
    """
    This function performs the PGD attack on the specified property characterised by its image, corresponding simplified
    model and epsilon value
    """
    # First, perturb the image randomly within the allowed bounds and perform a PGD attack
    lower_bound = torch.add(-epsilon * epsilon_factor, image)
    upper_bound = torch.add(epsilon * epsilon_factor, image)
    perturbation = torch.add(-epsilon * epsilon_factor,
                             2 * epsilon * epsilon_factor * torch.rand(image.size()))
    perturbed_image = torch.add(image, perturbation).clone().detach().requires_grad_(True)
    successful_attack_flag, heuristics_dict = gradient_ascent(simplified_model, perturbed_image, lower_bound,
                                                              upper_bound, pgd_learning_rate, num_iterations)

    # If the attack was successful, the procedure can be terminated and True can be returned
    if successful_attack_flag:
        return True

    # Otherwise, the GNN framework approach must be followed. First, initialise the GNN for the given network
    return  # TODO


def gradient_ascent(simplified_model, perturbed_image, lower_bound, upper_bound, pgd_learning_rate, num_iterations):
    """
    This function performs Gradient Ascent on the specified property given the bounds on the input and, if it didn't
    lead to positive loss, outputs the information about gradients
    """
    # Initialise the relevant optimiser and the list to store the gradients to be later used for heuristics generation
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
    gradient_magnitudes_pixels = grouped_by_epoch_to_grouped_by_pixel(gradient_magnitudes)

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
    mean_gradient_magnitude = torch.reshape(torch.tensor(mean_gradient_magnitudes_pixels), (1, 3, 32, 32))
    median_gradient_magnitude = torch.reshape(torch.tensor(median_gradient_magnitudes_pixels), (1, 3, 32, 32))
    max_gradient_magnitude = torch.reshape(torch.tensor(max_gradient_magnitudes_pixels), (1, 3, 32, 32))
    min_gradient_magnitude = torch.reshape(torch.tensor(min_gradient_magnitudes_pixels), (1, 3, 32, 32))
    std_gradient_magnitude = torch.reshape(torch.tensor(std_gradient_magnitudes_pixels), (1, 3, 32, 32))

    # Add the above statistics to the heuristics dictionary
    gradient_info_dict['mean gradient'] = mean_gradient_magnitude
    gradient_info_dict['median gradient'] = median_gradient_magnitude
    gradient_info_dict['maximum gradient'] = max_gradient_magnitude
    gradient_info_dict['minimum gradient'] = min_gradient_magnitude
    gradient_info_dict['gradient std'] = std_gradient_magnitude

    return gradient_info_dict


def grouped_by_epoch_to_grouped_by_pixel(gradients):
    """
    This function transforms the list of gradient magnitudes where each list element is the tensor represents the
    gradient at a particular epoch to the list where each element is the tensor representing the gradient of a
    particular pixel
    """
    gradients_grouped_by_pixels = []
    for matrix_idx in range(len(gradients[0][0])):
        for row_idx in range(len(gradients[0][0][0])):
            for value_idx in range(len(gradients[0][0][0][0])):
                gradients_grouped_by_pixels.append(torch.cat([torch.tensor([gradients[i][0][matrix_idx][row_idx]
                                                                            [value_idx].item()]) for i in
                                                              range(len(gradients))]))
    return gradients_grouped_by_pixels


pgd_gnn_attack_properties('base_easy.pkl', 'cifar_base_kw', 1, 0.1, 1, 10, subset=[0])
