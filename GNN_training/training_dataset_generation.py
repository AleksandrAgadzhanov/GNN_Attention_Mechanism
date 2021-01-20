import torch
from exp_utils.model_utils import load_verified_data, match_with_properties
from GNN_framework.helper_functions import match_with_subset, simplify_model
from GNN_training.pgd_branch_and_bound import pgd_branch_and_bound


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

        # Append the true and test labels at the end of each dictionary
        for j in range(len(list_of_feature_dicts)):
            # Also, if it is the first feature dictionary appended, include the image size in it
            if i == 0 and j == 0:
                list_of_feature_dicts[j]['input size'] = images[i].size()

            list_of_feature_dicts[j]['true label'] = true_labels[i]
            list_of_feature_dicts[j]['test label'] = test_labels[i]

        # Append all of the elements from the above list to the overall list of dictionaries
        for k in range(len(list_of_feature_dicts)):
            overall_list_of_feature_dicts.append(list_of_feature_dicts[k])

    # Store all the generated subdomains in a file
    torch.save(overall_list_of_feature_dicts, '../GNN_training/' + output_filename)


def main():
    generate_training_dataset('train_props.pkl', 'cifar_base_kw', 0.1, 10, 100, 'training_dataset.pkl', subset=[0])


if __name__ == '__main__':
    main()
