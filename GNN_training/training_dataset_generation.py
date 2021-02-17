import torch
import mlogger
from exp_utils.model_utils import load_verified_data, match_with_properties
from GNN_framework.helper_functions import match_with_subset, simplify_model
from GNN_training.helper_functions import pgd_attack_property_until_successful, pgd_attack_property_until_unsuccessful


def generate_training_dataset(properties_filename, model_name, pgd_learning_rate, num_iterations, output_filename,
                              epsilon_factor=1.0, subset=None, device='cpu'):
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

    # Go over one property at a time and make a function call which deals with the single property
    for i in range(len(images)):
        # First, simplify the network by adding the final layer and merging the last two layers into one, incorporating
        # the information about the true and test classes into the network
        simplified_model = simplify_model(model, true_labels[i], test_labels[i])

        # The first attack needs to be unsuccessful because to utilise each training property in training the GNN
        # feature vectors have to be constructed for each property and this cannot be achieved if there is no
        # information about the gradients of the previous unsuccessful PGD attack. Hence, perform the first attack
        # until it is unsuccessful
        feature_dict = pgd_attack_property_until_unsuccessful(simplified_model, images[i], epsilons[i] * epsilon_factor,
                                                              pgd_learning_rate, num_iterations, device=device)
        with mlogger.stdout_to('GNN_training/training_dataset_generation_log.txt'):
            print("Image " + str(i + 1) + " was attacked unsuccessfully")

        # Now make a call to the function which attacks the property until a successful counter-example is found in
        # order to obtain the ground-truth values of a successful attack
        ground_truth_attack = pgd_attack_property_until_successful(simplified_model, images[i], epsilons[i] *
                                                                   epsilon_factor, pgd_learning_rate, num_iterations,
                                                                   device=device)
        with mlogger.stdout_to('GNN_training/training_dataset_generation_log.txt'):
            print("Image " + str(i + 1) + " was attacked successfully")

        # Add the ground truth attack to the feature dictionary of the current property. Also add its true and test
        # labels to the dictionary
        feature_dict['successful attack'] = ground_truth_attack
        feature_dict['true label'] = true_labels[i]
        feature_dict['test label'] = test_labels[i]

        # Append the generated feature dictionary to the overall list
        overall_list_of_feature_dicts.append(feature_dict)

        # Store all the generated subdomains in a file
        torch.save(overall_list_of_feature_dicts, 'cifar_exp/' + output_filename)


def main():
    generate_training_dataset('train_SAT_med.pkl', 'cifar_base_kw', 0.001, 50000, 'train_SAT_med_dataset.pkl',
                              device='cuda')


if __name__ == '__main__':
    main()
