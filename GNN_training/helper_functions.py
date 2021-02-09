import torch
from GNN_framework.features_generation import generate_input_feature_vectors, generate_relu_output_feature_vectors
from GNN_framework.helper_functions import perturb_image, gradient_ascent


def generate_feature_dict(neural_network, lower_bound, upper_bound, image, perturbed_image, epsilon,
                          gradient_info_dict):
    """
    This function generates a complete feature dictionary for one a subdomain of the training dataset. This dictionary
    consists of the input feature vectors, list of feature vectors of the ReLU nodes and output feature vectors.
    """
    # Use the special function to generate all the input feature vectors
    input_feature_vectors = generate_input_feature_vectors(lower_bound, upper_bound, perturbed_image,
                                                           gradient_info_dict)

    # Use the special function to generate all the hidden layer and output feature vectors
    relu_feature_vectors_list, output_feature_vectors = generate_relu_output_feature_vectors(neural_network,
                                                                                             lower_bound, upper_bound,
                                                                                             image, perturbed_image,
                                                                                             epsilon)

    # Construct the feature dictionary and return it
    feature_dict = {'input': input_feature_vectors, 'hidden': relu_feature_vectors_list,
                    'output': output_feature_vectors}

    return feature_dict


# TODO create a better method based on more heuristics than just last gradient magnitude
def compute_pixel_scores(feature_dict):
    """
    This function takes a feature dictionary of a particular subdomain and computes the scores for all pixels based on
    the information contained in the gradient information part of the dictionary. In this implementation, the scores are
    based only on the last gradient magnitudes of pixels.
    """
    # Initialise the output scores variable as the tensor of last gradient magnitudes of pixels
    pixel_scores = feature_dict['input'][3, :]

    return pixel_scores


def pgd_attack_property_until_successful(simplified_model, image, epsilon, pgd_learning_rate, num_iterations):
    """
    This function performs randomly initialized PGD attacks on a given property until a counterexample is found. When
    it happens, the function returns the pixel values which resulted in a successful attack.
    """
    # Initialize the variable which will be indicating whether the property was successfully attacked and also the
    # variable storing the original epsilon in case it has to be increased by a constant amount
    successful_attack_flag = False
    original_epsilon = epsilon

    # Now perform PGD attacks on a given property until a counter-example is found
    while not successful_attack_flag:
        # Initialize a random PGD attack
        lower_bound = torch.add(-epsilon, image)
        upper_bound = torch.add(epsilon, image)
        perturbed_image = perturb_image(lower_bound, upper_bound)

        # Perform gradient ascent on the PGD attack initialized above
        successful_attack_flag, perturbed_image, _ = gradient_ascent(simplified_model, perturbed_image, lower_bound,
                                                                     upper_bound, pgd_learning_rate, num_iterations)

        # If the attack was successful, return the last values of the perturbed pixels
        if successful_attack_flag:
            return perturbed_image

        # Otherwise, increase the epsilon factor by 1%
        epsilon += 0.01 * original_epsilon


def pgd_attack_property_until_unsuccessful(simplified_model, image, epsilon, pgd_learning_rate, num_iterations):
    """
    This function performs randomly initialized PGD attacks on a given property until one of them is unsuccessful. When
    it happens, the function returns the feature dictionary associated with the unsuccessful attack.
    """
    # Initialize the variable which will be indicating whether the property was successfully attacked and also the
    # variable storing the original epsilon in case it has to be decreased by a constant amount
    successful_attack_flag = True
    original_epsilon = epsilon

    # Now perform PGD attacks on a given property until one of the, is unsuccessful
    while successful_attack_flag:
        # Initialize a random PGD attack
        lower_bound = torch.add(-epsilon, image)
        upper_bound = torch.add(epsilon, image)
        perturbed_image = perturb_image(lower_bound, upper_bound)

        # Perform gradient ascent on the PGD attack initialized above
        successful_attack_flag, perturbed_image, gradient_info_dict = gradient_ascent(simplified_model, perturbed_image,
                                                                                      lower_bound, upper_bound,
                                                                                      pgd_learning_rate, num_iterations)

        # If the attack was unsuccessful, return the gradient information dictionary
        if not successful_attack_flag:
            feature_dict = generate_feature_dict(simplified_model, lower_bound, upper_bound, image, perturbed_image,
                                                 epsilon, gradient_info_dict)
            return feature_dict

        # Otherwise, decrease the epsilon factor by 1%
        epsilon -= 0.01 * original_epsilon


def compute_loss(new_lower_bound, new_upper_bound, ground_truth_attack, loss_lambda):
    """
    This function computes the loss characterised by the bounds output from the GNN and the ground truth PGD attack
    pixel values from the training dataset. It does so by using a convex approximation to the 0-1 loss associated with
    whether the ground truth PGD attack lies within the bounds produced by the GNN and also penalising the GNN for not
    narrowing down the search space.
    """
    # Evaluate the first loss term for all pixels. If the ground truth pixel value lies within the bounds produced by
    # the GNN, set the loss associated with it to zero. If it lies below the lower bound produced by the GNN, return the
    # difference between the lower bound and the pixel value. Finally, if it lies above the upper bound produced by the
    # GNN, return the difference between the pixel value and the upper bound.
    loss_term_1_pixels = torch.where(new_lower_bound - ground_truth_attack > 0,
                                     torch.add(new_lower_bound, -ground_truth_attack),
                                     torch.where(ground_truth_attack - new_upper_bound > 0,
                                                 torch.add(ground_truth_attack, -new_upper_bound),
                                                 torch.zeros(new_lower_bound.size())))

    # Determine the average pixel-wise loss for the first term
    num_pixels = new_lower_bound.view(-1).size()[0]
    loss_term_1 = torch.sum(loss_term_1_pixels) / num_pixels

    # Evaluate the second loss term for all pixels. This is simply the average of the sum of differences of the upper
    # and lower bounds for each pixel.
    loss_term_2 = torch.sum(torch.add(new_upper_bound, -new_lower_bound)) / num_pixels

    # Return the overall loss which is simply the sum of the two terms computed above
    loss = loss_term_1 + loss_lambda * loss_term_2

    return loss
