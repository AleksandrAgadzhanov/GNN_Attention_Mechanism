import time
from exp_utils.model_utils import load_verified_data, match_with_properties
from GNN_framework.helper_functions import match_with_subset, simplify_model


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

        subdomain_lower_bounds, subdomain_upper_bounds = pgd_branch_and_bound(simplified_model, images[i], epsilons[i],
                                                                              pgd_learning_rate, num_iterations, timeout
                                                                              )


def pgd_branch_and_bound(simplified_model, image, epsilon, pgd_learning_rate, num_iterations, timeout):
    """
    This function tries to successfully perform a PGD attack on a given image by following the Branch & Bound algorithm,
    generating subdomains in the process for the training dataset which it then returns as lists of lower and upper
    bounds.
    """
    pass


def main():
    # print(generate_training_dataset('train_props.pkl', 'cifar_base_kw'))
    pass


if __name__ == '__main__':
    main()
