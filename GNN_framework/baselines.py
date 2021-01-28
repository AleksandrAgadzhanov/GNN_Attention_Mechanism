import torch
from exp_utils.model_utils import load_verified_data, match_with_properties
from helper_functions import match_with_subset, simplify_model, perturb_image, gradient_ascent


def pgd_attack_properties(properties_filename, model_name, epsilon_factor, pgd_learning_rate, num_iterations,
                          subset=None):
    """
    This function acts as the 1st baseline to compare the pgd_gnn_attack_property() function against. It simply
    initialises a single PGD attack randomly and performs a specified number of iterations of gradient ascent. For
    effective comparison, the total number of iterations in this function must match the total number of iterations made
    by the pgd_gnn_attack_property() function, given by number_of_epochs * number_of_iterations_per_epoch.
    """
    # Load all the required data for the images which were correctly verified by the model
    verified_images, verified_true_labels, verified_image_indices, model = load_verified_data(model_name)

    # Load the lists of images, true and test labels and epsilons for images which appear in both the properties dataset
    # as verified and in the previously loaded images list
    images, true_labels, test_labels, epsilons = match_with_properties(properties_filename, verified_images,
                                                                       verified_true_labels, verified_image_indices)

    # If the subset of indices was specified for the purpose of reducing the time complexity, drop the elements of
    # images, true_labels, test_labels and epsilons not indicated in the subset indices
    if subset is not None:
        images, true_labels, test_labels, epsilons = match_with_subset(subset, images, true_labels, test_labels,
                                                                       epsilons)

    # Now attack each property in turn by calling the appropriate function
    num_properties_still_verified = 0  # counter of properties which are still verified after the PGD attack
    for i in range(len(images)):
        # First, simplify the network by adding the final layer and merging the last two layers into one, incorporating
        # the information about the true and test classes into the network
        simplified_model = simplify_model(model, true_labels[i], test_labels[i])

        # First, perturb the image randomly within the allowed bounds
        lower_bound = torch.add(-epsilons[i] * epsilon_factor, images[i])
        upper_bound = torch.add(epsilons[i] * epsilon_factor, images[i])
        perturbed_image = perturb_image(lower_bound, upper_bound).requires_grad_(True)

        # Now perform a single PGD attack
        successful_attack_flag, _ = gradient_ascent(simplified_model, perturbed_image, lower_bound, upper_bound,
                                                    pgd_learning_rate, num_iterations)

        # If the attack was unsuccessful, increase the counter
        if not successful_attack_flag:
            num_properties_still_verified += 1

    # Calculate the verification accuracy for the properties in the file provided after all the PGD attacks
    verification_accuracy = 100.0 * num_properties_still_verified / len(images)

    return verification_accuracy


def pgd_attack_properties_trials(properties_filename, model_name, epsilon_factor, pgd_learning_rate, num_iterations,
                                 num_trials, subset=None):
    """
    This function acts as the 2nd baseline to compare the pgd_gnn_attack_property() function against. It initialises a
    specified number of trial random PGD attacks and performs a specified number of iterations of gradient ascent. For
    effective comparison, the product number_of_trials * number_of_iterations_per_trial must match the total number of
    iterations made by pgd_gnn_attack_property(), given by number_of_epochs * number_of_iterations_per_epoch.
    """
    # Load all the required data for the images which were correctly verified by the model
    verified_images, verified_true_labels, verified_image_indices, model = load_verified_data(model_name)

    # Load the lists of images, true and test labels and epsilons for images which appear in both the properties dataset
    # as verified and in the previously loaded images list
    images, true_labels, test_labels, epsilons = match_with_properties(properties_filename, verified_images,
                                                                       verified_true_labels, verified_image_indices)

    # If the subset of indices was specified for the purpose of reducing the time complexity, drop the elements of
    # images, true_labels, test_labels and epsilons not indicated in the subset indices
    if subset is not None:
        images, true_labels, test_labels, epsilons = match_with_subset(subset, images, true_labels, test_labels,
                                                                       epsilons)

    # Now attack each property in turn for the specified number of trials
    num_properties_still_verified = 0  # counter of properties which are still verified after the PGD attack
    for i in range(len(images)):
        # First, simplify the network by adding the final layer and merging the last two layers into one,
        # incorporating the information about the true and test classes into the network
        simplified_model = simplify_model(model, true_labels[i], test_labels[i])

        for j in range(num_trials):
            # First, perturb the image randomly within the allowed bounds
            lower_bound = torch.add(-epsilons[i] * epsilon_factor, images[i])
            upper_bound = torch.add(epsilons[i] * epsilon_factor, images[i])
            perturbed_image = perturb_image(lower_bound, upper_bound).requires_grad_(True)

            # Now perform a single PGD attack
            successful_attack_flag, _ = gradient_ascent(simplified_model, perturbed_image, lower_bound, upper_bound,
                                                        pgd_learning_rate, num_iterations)

            # If the attack was unsuccessful, increase the counter and break from the loop
            if not successful_attack_flag:
                num_properties_still_verified += 1
                break

    # Calculate the verification accuracy for the properties in the file provided after all the PGD attacks
    verification_accuracy = 100.0 * num_properties_still_verified / len(images)

    return verification_accuracy
