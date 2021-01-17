import time
import torch
import copy
from exp_utils.model_utils import load_verified_data, match_with_properties
from GNN_framework.helper_functions import match_with_subset, simplify_model, perturb_image, gradient_ascent


def generate_training_dataset(properties_filename, model_name, pgd_learning_rate, num_iterations, timeout, subset=None):
    """
    This function generates the training dataset by going through the provided property dataset and applying Branch &
    Bound algorithm to each property while storing all the considered subdomains which will comprise the training
    dataset.
    """
    # Load all the required data for the images which were correctly verified by the model
    correctly_verified_images, true_labels_cv_images, image_indices_cv_images, model = \
        load_verified_data(model_name)

    # Load the lists of images, true and test labels and epsilons for images which appear in both the properties dataset
    # as verified and in the previously loaded images list
    images, true_labels, test_labels, epsilons = match_with_properties(properties_filename, correctly_verified_images,
                                                                       true_labels_cv_images, image_indices_cv_images)

    # If the subset of indices was specified for the purpose of reducing the time complexity, drop the elements of
    # images, true_labels, test_labels and epsilons not indicated in the subset indices
    if subset is not None:
        images, true_labels, test_labels, epsilons = match_with_subset(subset, images, true_labels, test_labels,
                                                                       epsilons)

    # TODO Initialise the storage

    # Now go over one property at a time and perform Branch & Bound algorithm on it with a specified timeout to obtain a
    # set of subdomains
    for i in range(len(images)):
        # First, simplify the network by adding the final layer and merging the last two layers into one, incorporating
        # the information about the true and test classes into the network
        simplified_model = simplify_model(model, true_labels[i], test_labels[i])

        lower_bounds_list, upper_bounds_list = pgd_branch_and_bound(simplified_model, images[i], epsilons[i],
                                                                    pgd_learning_rate, num_iterations, timeout)

        # TODO append the bounds to storage

    # TODO store all the generated subdomain bounds


def pgd_branch_and_bound(simplified_model, image, epsilon, pgd_learning_rate, num_iterations, timeout):
    """
    This function tries to successfully perform a PGD attack on a given image by following the Branch & Bound algorithm,
    generating subdomains in the process for the training dataset which it then returns as lists of lower and upper
    bounds.
    """
    # Initialise the storage variables for lower and upper bounds which will represent the candidate domains to branch
    # on, with the first and for now only elements being the lower and upper bounds of the original domain. Also
    # initialise the list where the respective subdomain heuristics will be stored (in row tensor form)
    lower_bounds_list = [torch.add(image, -epsilon)]
    upper_bounds_list = [torch.add(image, epsilon)]
    heuristics_list = []

    # Initialise the storage variables for lower and upper bounds of all considered subdomains which will be the outputs
    lower_bounds_output_list = copy.deepcopy(lower_bounds_list)
    upper_bounds_output_list = copy.deepcopy(upper_bounds_list)

    # Perturb the original image randomly within the bounds above
    perturbed_image = perturb_image(lower_bounds_list[0], upper_bounds_list[0])

    # Run the first PGD attack on the original domain and store the heuristics to append them to the storage
    successful_attack_flag, gradient_info_dict = gradient_ascent(simplified_model, perturbed_image,
                                                                 lower_bounds_list[0], upper_bounds_list[0],
                                                                 pgd_learning_rate, num_iterations)
    heuristics_list.append(gradient_info_dict['mean gradient'].view(-1))

    # If the attack was successful, then the only subdomain to be returned is the original one
    if successful_attack_flag:
        return lower_bounds_output_list, upper_bounds_output_list

    # Otherwise, start the timer
    start_time = time.time()

    # Now begin the Branch & Bound algorithm based on Algorithm 1 of the "NN Branching for NN Verification" paper
    while time.time() - start_time < timeout:
        # Pick out the subdomain on which branching will be performed
        best_subdomain_index, best_pixel_index = pick_out(heuristics_list)



    return lower_bounds_output_list, upper_bounds_output_list


def pick_out(heuristics_list):
    """
    This function picks the subdomain and the input node within this subdomain on which further branching will be
    performed based on the list of heuristics of different subdomains. It then returns the index of the required
    subdomain and the index of the required input node.
    """
    # Initialise the variable storing the maximum mean gradient magnitude over all the subdomains, the associated input
    # node index and subdomain index
    max_max_mean_gradient_magnitude = 0
    best_pixel_index = 0
    best_subdomain_index = 0

    # Go through the heuristics list and search for the best input node to branch on
    for subdomain_idx in range(len(heuristics_list)):
        max_mean_gradient_magnitude, max_mean_gradient_magnitude_idx = torch.max(heuristics_list[subdomain_idx], dim=0)
        if max_mean_gradient_magnitude > max_max_mean_gradient_magnitude:
            max_max_mean_gradient_magnitude = max_mean_gradient_magnitude
            best_subdomain_index = subdomain_idx
            best_pixel_index = max_mean_gradient_magnitude_idx

    return best_subdomain_index, best_pixel_index


def main():
    #print(generate_training_dataset('train_props.pkl', 'cifar_base_kw'))
    x = torch.rand(5)
    print(x)
    max, max_idx = torch.max(x, dim=0)
    print(max)
    print(max_idx)


if __name__ == '__main__':
    main()
