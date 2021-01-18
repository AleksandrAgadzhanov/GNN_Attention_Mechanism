import time
import torch
import copy
from exp_utils.model_utils import load_verified_data, match_with_properties
from GNN_framework.helper_functions import match_with_subset, simplify_model, perturb_image, gradient_ascent


def generate_training_dataset(properties_filename, model_name, pgd_learning_rate, num_iterations, timeout,
                              output_filename, subset=None):
    """
    This function generates the training dataset by going through the provided property dataset and applying Branch &
    Bound algorithm to each property while storing all the considered subdomains and their input features which will
    comprise the training dataset.
    """
    # Load all the required data for the images which were correctly verified by the model
    correctly_verified_images, true_labels_cv_images, image_indices_cv_images, model = load_verified_data(model_name)

    # Load the lists of images, true and test labels and epsilons for images which appear in both the properties dataset
    # as verified and in the previously loaded images list
    images, true_labels, test_labels, epsilons = match_with_properties(properties_filename, correctly_verified_images,
                                                                       true_labels_cv_images, image_indices_cv_images)

    # If the subset of indices was specified for the purpose of reducing the time complexity, drop the elements of
    # images, true_labels, test_labels and epsilons not indicated in the subset indices
    if subset is not None:
        images, true_labels, test_labels, epsilons = match_with_subset(subset, images, true_labels, test_labels,
                                                                       epsilons)

    # Initialise the overall list of dictionaries for storing the input features of all the considered subdomains
    overall_list_of_feature_dicts = []

    # Now go over one property at a time and perform Branch & Bound algorithm on it
    for i in range(len(images)):
        # First, simplify the network by adding the final layer and merging the last two layers into one, incorporating
        # the information about the true and test classes into the network
        simplified_model = simplify_model(model, true_labels[i], test_labels[i])

        # Apply the Branch & Bound algorithm and store the resulting list of dictionaries of input features
        list_of_feature_dicts = pgd_branch_and_bound(simplified_model, images[i], epsilons[i], pgd_learning_rate,
                                                     num_iterations, timeout)

        # Append all of the elements from the above list to the overall list of dictionaries
        for j in range(len(list_of_feature_dicts)):
            overall_list_of_feature_dicts.append(list_of_feature_dicts[j])

    # Store all the generated subdomains in a file
    torch.save(overall_list_of_feature_dicts, '../GNN_training/' + output_filename)


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
    feature_dict = generate_feature_dict(lower_bound, upper_bound, perturbed_image, gradient_info_dict)
    list_of_feature_dicts.append(feature_dict)

    # If the attack was successful, then the only subdomain to be returned is the original one
    if successful_attack_flag:
        return list_of_feature_dicts

    # Otherwise initialise the list of subdomains suitable for branching and start the timer
    list_of_feature_dicts_branch = copy.deepcopy(list_of_feature_dicts)
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
            lower_bound = list_of_feature_dicts_branch[subdomain_index]['lower bound']
            upper_bound = list_of_feature_dicts_branch[subdomain_index]['upper bound']
            perturbed_image = list_of_feature_dicts_branch[subdomain_index]['perturbed image'].requires_grad_(True)
            successful_attack_flag, gradient_info_dict = gradient_ascent(simplified_model, perturbed_image, lower_bound,
                                                                         upper_bound, pgd_learning_rate, num_iterations)
            feature_dict = generate_feature_dict(lower_bound, upper_bound, perturbed_image, gradient_info_dict)
            list_of_feature_dicts.append(feature_dict)

            # If the PGD attack was successful, return the list of the subdomains
            if successful_attack_flag:
                return list_of_feature_dicts

            # Otherwise, update the gradient information part of the dictionary of the new subdomain which was attacked
            list_of_feature_dicts_branch[subdomain_index] = feature_dict

    # If this point was reached, then all the PGD attacks were unsuccessful within the time allowed, so just return all
    # the generated subdomains anyway

    return list_of_feature_dicts


def generate_feature_dict(lower_bound, upper_bound, perturbed_image, gradient_info_dict):
    """
    This function accepts tensors of lower and upper bounds and perturbed image values together with the gradient
    information dictionary and constructs the dictionary of features out of this tensor, where each value of the
    dictionary has the same shape as the image.
    """
    # Initialise the output dictionary with the first 3 key-value relationships
    feature_dict = {'lower bound': lower_bound,
                    'upper bound': upper_bound,
                    'perturbed image': perturbed_image}

    # Reshape each tensor and list in the gradient info dictionary to be of the image shape to be consistent
    for key in gradient_info_dict.keys():
        if type(gradient_info_dict[key]) == list:
            gradient_info_dict[key] = torch.tensor(gradient_info_dict[key]).reshape(lower_bound.size())
        else:
            gradient_info_dict[key] = gradient_info_dict[key].reshape(lower_bound.size())

    # Merge the above dictionary with the gradient info dictionary and return it
    feature_dict.update(gradient_info_dict)

    return feature_dict


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


# TODO create a better method based on more heuristics than just last gradient magnitude
def compute_pixel_scores(feature_dict):
    """
    This function takes a feature dictionary of a particular subdomain and computes the scores for all pixels based on
    the information contained in the gradient information part of the dictionary. In this implementation, the scores are
    based only on the last gradient magnitudes of pixels.
    """
    # Initialise the output scores variable
    pixel_scores = feature_dict['last gradient'].view(-1)

    return pixel_scores


def split(list_of_feature_dicts, subdomain_index, pixel_index):
    """
    This function accepts a list of subdomains characterised by their feature dictionaries and makes a split on a
    particular input node of a particular subdomain, appending two new subdomains to and removing the original one from
    the list of all subdomains. The split is made in the middle of the selected node values range.
    """
    # First, remove the required subdomain from the list and store its feature dictionary for now
    removed_feature_dict = list_of_feature_dicts.pop(subdomain_index)

    # Store the original image size for later reshaping and retrieve the original lower and upper bound of the domain on
    # which the splitting will be performed
    image_size = removed_feature_dict['lower bound'].size()
    old_lower_bound = removed_feature_dict['lower bound'].view(-1)
    old_upper_bound = removed_feature_dict['upper bound'].view(-1)

    # Initialise the new upper bound of the new "left" domain
    new_upper_bound_left = copy.deepcopy(old_upper_bound)
    new_upper_bound_left[pixel_index] = 0.5 * (old_upper_bound[pixel_index] + old_upper_bound[pixel_index])
    new_upper_bound_left = new_upper_bound_left.reshape(image_size)

    # And the new lower bound of the new "right" domain
    new_lower_bound_right = copy.deepcopy(old_lower_bound)
    new_lower_bound_right[pixel_index] = 0.5 * (old_upper_bound[pixel_index] + old_upper_bound[pixel_index])
    new_lower_bound_right = new_lower_bound_right.reshape(image_size)

    # Make two copies of the dictionary of the original domain and set the lower and upper bounds to the appropriate
    # values. Then append these two new dictionaries to the subdomain list
    new_feature_dict_left = copy.deepcopy(removed_feature_dict)
    new_feature_dict_right = copy.deepcopy(removed_feature_dict)
    new_feature_dict_left['upper bound'] = new_upper_bound_left
    new_feature_dict_right['lower bound'] = new_lower_bound_right
    list_of_feature_dicts.append(new_feature_dict_left)
    list_of_feature_dicts.append(new_feature_dict_right)

    return list_of_feature_dicts


def main():
    print(generate_training_dataset('train_props.pkl', 'cifar_base_kw', 0.1, 10, 100, 'training_dataset.pkl',
                                    subset=[0]))


if __name__ == '__main__':
    main()
