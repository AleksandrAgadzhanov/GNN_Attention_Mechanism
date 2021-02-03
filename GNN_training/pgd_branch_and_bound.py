import torch
import time
from GNN_framework.helper_functions import perturb_image, gradient_ascent
from GNN_training.helper_functions import generate_feature_dict, compute_pixel_scores


def pgd_branch_and_bound(simplified_model, image, epsilon, pgd_learning_rate, num_iterations, timeout):
    """
    This function tries to successfully perform a PGD attack on a given image by following the Branch & Bound algorithm,
    generating subdomains in the process for the training dataset which it then returns as a list of dictionaries
    containing input features information about each considered subdomain.
    """
    # Initialise the list of dictionaries which will contain information about each subdomain
    list_of_feature_dicts = []

    # Compute the initial lower and upper bounds and perturb the image for the first PGD attack
    lower_bound = torch.add(-epsilon, image)
    upper_bound = torch.add(epsilon, image)
    perturbed_image = perturb_image(lower_bound, upper_bound).requires_grad_(True)

    # Run the first PGD attack on the original domain and store the heuristics to append them to the storage later
    successful_attack_flag, gradient_info_dict = gradient_ascent(simplified_model, perturbed_image, lower_bound,
                                                                 upper_bound, pgd_learning_rate, num_iterations)

    # Construct the first dictionary to be appended to the output list by using a special function
    feature_dict = generate_feature_dict(simplified_model, lower_bound, upper_bound, image, perturbed_image, epsilon,
                                         gradient_info_dict)
    list_of_feature_dicts.append(feature_dict)

    # If the attack was successful, then the only subdomain to be returned is the original one
    if successful_attack_flag:
        return list_of_feature_dicts

    # Otherwise initialise the list of subdomains suitable for branching and start the timer
    list_of_feature_dicts_branch = list_of_feature_dicts.copy()
    start_time = time.time()

    # Now begin the Branch & Bound algorithm based on Algorithm 1 of the "NN Branching for NN Verification" paper
    while time.time() - start_time < timeout:
        # Pick out the subdomain on which branching will be performed
        best_subdomain_index, best_pixel_index = pick_out(list_of_feature_dicts_branch)

        # Make a split on the chosen pixel of the chosen subdomain by using a special function
        list_of_feature_dicts_branch = split(list_of_feature_dicts_branch, best_subdomain_index, best_pixel_index)

        # Now perform the PGD attacks on the two new subdomains by using the same approach as above
        for subdomain_index in range(len(list_of_feature_dicts_branch) - 2,
                                     len(list_of_feature_dicts_branch), 1):
            lower_bound = list_of_feature_dicts_branch[subdomain_index]['input'][0, :].reshape(image.size())
            upper_bound = list_of_feature_dicts_branch[subdomain_index]['input'][1, :].reshape(image.size())
            perturbed_image = list_of_feature_dicts_branch[subdomain_index]['input'][2, :].reshape(image.size())
            perturbed_image.requires_grad_(True)
            successful_attack_flag, gradient_info_dict = gradient_ascent(simplified_model, perturbed_image, lower_bound,
                                                                         upper_bound, pgd_learning_rate, num_iterations)
            feature_dict = generate_feature_dict(simplified_model, lower_bound, upper_bound, image, perturbed_image,
                                                 epsilon, gradient_info_dict)
            list_of_feature_dicts.append(feature_dict)

            # If the PGD attack was successful, return the list of the subdomains
            if successful_attack_flag:
                return list_of_feature_dicts

            # Otherwise, update the gradient information part of the dictionary of the new subdomain which was attacked
            list_of_feature_dicts_branch[subdomain_index] = feature_dict

    # If this point was reached, then all the PGD attacks were unsuccessful within the time allowed, so just return all
    # the generated subdomains anyway

    return list_of_feature_dicts


def pick_out(list_of_feature_dicts):
    """
    This function picks the subdomain and the input node within this subdomain on which further branching will be
    performed based on the list of heuristics of different subdomains. It then returns the index of the required
    subdomain and the index of the required input node.
    """
    # Initialise the indices of the best subdomain to branch on and of the best input node to branch on within this
    # subdomain. Also initialise the variable which will be storing the best score of the individual pixel
    best_subdomain_index = 0
    best_pixel_index = 0
    best_score = 0

    # Go over all subdomains in turn
    for subdomain_index in range(len(list_of_feature_dicts)):
        # Find which pixel in the current subdomain has the best score and store this score and its index
        feature_dict = list_of_feature_dicts[subdomain_index]
        pixel_scores = compute_pixel_scores(feature_dict)
        max_score, max_score_pixel_index = torch.max(pixel_scores, dim=0)

        # If the maximum score of the pixel in this subdomain is larger than the global maximum, update the variables
        if max_score > best_score:
            best_score = max_score
            best_pixel_index = max_score_pixel_index
            best_subdomain_index = subdomain_index

    return best_subdomain_index, best_pixel_index


def split(list_of_feature_dicts, subdomain_index, pixel_index):
    """
    This function accepts a list of subdomains characterised by their feature dictionaries and makes a split on a
    particular input node of a particular subdomain, appending two new subdomains to and removing the original one from
    the list of all subdomains. The split is made in the middle of the selected node values range.
    """
    # First, remove the required subdomain from the list and store its feature dictionary for now
    removed_feature_dict = list_of_feature_dicts.pop(subdomain_index)

    # Retrieve the original lower and upper bound of the domain on which the splitting will be performed
    old_lower_bound = removed_feature_dict['input'][0, :]
    old_upper_bound = removed_feature_dict['input'][1, :]

    # Initialise the new upper bound of the new "left" domain
    new_upper_bound_left = old_upper_bound.clone().detach()
    new_upper_bound_left[pixel_index] = 0.5 * (old_upper_bound[pixel_index] + old_upper_bound[pixel_index])

    # And the new lower bound of the new "right" domain
    new_lower_bound_right = old_lower_bound.clone().detach()
    new_lower_bound_right[pixel_index] = 0.5 * (old_upper_bound[pixel_index] + old_upper_bound[pixel_index])

    # Make two copies of the dictionary of the original domain and set the lower and upper bounds to the appropriate
    # values. Then append these two new dictionaries to the subdomain list
    new_feature_dict_left = removed_feature_dict.copy()
    new_feature_dict_right = removed_feature_dict.copy()
    new_feature_dict_left['input'][1, :] = new_upper_bound_left
    new_feature_dict_right['input'][0, :] = new_lower_bound_right
    list_of_feature_dicts.append(new_feature_dict_left)
    list_of_feature_dicts.append(new_feature_dict_right)

    return list_of_feature_dicts
