import torch
from exp_utils.model_utils import load_verified_data, match_with_properties
from GNN_framework.helper_functions import match_with_subset, simplify_model
from GNN_framework.helper_functions import perturb_image, gradient_ascent
from GNN_training.pgd_branch_and_bound import generate_feature_dict


def generate_training_dataset(properties_filename, model_name, pgd_learning_rate, num_iterations, output_filename,
                              epsilon_factor=1.0, subset=None):
    """
    This function generates the training dataset to perform supervised learning for the GNN. It does so by performing
    PGD attacks with random initializations and big number of steps, following Branch & Bound algorithm, until an
    adversarial example is found for each property. It then stores the lower and upper bounds associated with each such
    adversarial example in the training dataset. It acts as a wrapper for the pgd_attack_property_until_successful()
    function which deals with 1 property at a time.
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

    # Initialise the overall list of dictionaries for storing the features of all the considered properties
    overall_list_of_feature_dicts = []

    # Initialize the variable which will be keeping track of whether the lat PGD attack was successful or not
    successful_attack_flag = True

    # Go over one property at a time and make a function call which deals with the single property
    for i in range(len(images)):
        # First, simplify the network by adding the final layer and merging the last two layers into one, incorporating
        # the information about the true and test classes into the network
        simplified_model = simplify_model(model, true_labels[i], test_labels[i])

        # Now compute the lower and upper bounds and initialize the variables storing the perturbed image pixel values
        # and gradient information dictionary
        lower_bound = torch.add(-epsilons[i] * epsilon_factor, images[i])
        upper_bound = torch.add(epsilons[i] * epsilon_factor, images[i])
        perturbed_image = torch.zeros(lower_bound.size())
        gradient_info_dict = {}

        # The first attack needs to be unsuccessful because to utilise each training property in training the GNN
        # feature vectors have to be constructed for each property and this cannot be achieved if there is no
        # information about the gradients of the previous unsuccessful PGD attack. Hence, perform the first attack
        # until it is unsuccessful
        while successful_attack_flag:
            perturbed_image = perturb_image(lower_bound, upper_bound)
            successful_attack_flag, perturbed_image, gradient_info_dict = gradient_ascent(simplified_model,
                                                                                          perturbed_image, lower_bound,
                                                                                          upper_bound,
                                                                                          pgd_learning_rate,
                                                                                          num_iterations)

        # Generate the features dictionary for the current property by calling the appropriate function
        feature_dict = generate_feature_dict(simplified_model, lower_bound, upper_bound, images[i], perturbed_image,
                                             epsilons[i] * epsilon_factor, gradient_info_dict)

        # Now make a call to the function which attacks the property until a successful counter-example is found in
        # order to obtain the ground-truth values of a successful attack
        ground_truth_attack = pgd_attack_property_until_successful(simplified_model, images[i], epsilons[i] *
                                                                   epsilon_factor, pgd_learning_rate, num_iterations)

        # Add the ground truth attack to the feature dictionary of the current property
        feature_dict['successful attack'] = ground_truth_attack

        # Append the generated feature dictionary to the overall list
        overall_list_of_feature_dicts.append(feature_dict)

    # Store all the generated subdomains in a file
    torch.save(overall_list_of_feature_dicts, '../GNN_training/' + output_filename)


def pgd_attack_property_until_successful(simplified_model, image, epsilon, pgd_learning_rate, num_iterations):
    """
    This function performs randomly initialized PGD attacks on a given property until a counterexample is found. When
    it happens, the function returns the feature dictionary containing the correct
    """
    # Initialize the variable which will be indicating whether the property was successfully attacked
    successful_attack_flag = False

    # Compute the lower and upper bound within which the image pixels can be perturbed (these will be the same for all
    # PGD attacks)
    lower_bound = torch.add(-epsilon, image)
    upper_bound = torch.add(epsilon, image)

    # Now perform PGD attacks on a given property until a counter-example is found
    while not successful_attack_flag:
        # Initialize a random PGD attack
        perturbed_image = perturb_image(lower_bound, upper_bound)

        # Perform gradient ascent on the PGD attack initialized above
        successful_attack_flag, perturbed_image, _ = gradient_ascent(simplified_model, perturbed_image, lower_bound,
                                                                     upper_bound, pgd_learning_rate, num_iterations)

        # If the attack was successful, return the last values of the perturbed pixels
        if successful_attack_flag:
            return perturbed_image


def main():
    generate_training_dataset('train_SAT_med.pkl', 'cifar_base_kw', 0.1, 100, 'training_dataset.pkl', subset=[0], epsilon_factor=1)


if __name__ == '__main__':
    main()
