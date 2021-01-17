from exp_utils.model_utils import load_verified_data, match_with_properties
from GNN_framework.GraphNeuralNetwork import GraphNeuralNetwork
from GNN_framework.helper_functions import match_with_subset, simplify_model, perturb_image, gradient_ascent, \
    update_domain_bounds
from GNN_framework.features_generation import generate_input_feature_vectors, generate_relu_output_feature_vectors
import torch


def pgd_gnn_attack_properties(properties_filename, model_name, epsilon_factor, pgd_learning_rate, num_iterations,
                              num_epochs, num_updates, embedding_vector_size, auxiliary_hidden_size,
                              num_update_methods=3, subset=None):
    """
    This function acts aims to find adversarial examples for each property in the file specified. It acts as a container
    for the function which attacks each property in turn by calling this function for each property.
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

        successful_attack_flag = pgd_gnn_attack_property(simplified_model, images[i], epsilons[i], epsilon_factor,
                                                         pgd_learning_rate, num_iterations, num_epochs, num_updates,
                                                         embedding_vector_size, auxiliary_hidden_size,
                                                         num_update_methods)

        # If the attack was unsuccessful, increase the counter
        if not successful_attack_flag:
            num_properties_still_verified += 1

    # Calculate the verification accuracy for the properties in the file provided after all the PGD attacks
    verification_accuracy = 100.0 * num_properties_still_verified / len(images)

    return verification_accuracy


def pgd_gnn_attack_property(simplified_model, image, epsilon, epsilon_factor, pgd_learning_rate, num_iterations,
                            num_epochs, num_updates, embedding_vector_size, auxiliary_hidden_size,
                            num_update_methods):
    """
    This function performs the PGD attack on the specified property characterised by its image, corresponding simplified
    model and epsilon value
    """
    # First, perturb the image randomly within the allowed bounds and perform a PGD attack
    lower_bound = torch.add(-epsilon * epsilon_factor, image)
    upper_bound = torch.add(epsilon * epsilon_factor, image)
    perturbed_image = perturb_image(lower_bound, upper_bound).requires_grad_(True)
    successful_attack_flag, gradient_info_dict = gradient_ascent(simplified_model, perturbed_image, lower_bound,
                                                                 upper_bound, pgd_learning_rate, num_iterations)

    # If the attack was successful, the procedure can be terminated and True can be returned
    if successful_attack_flag:
        return True

    # Otherwise, the GNN framework approach must be followed. First, generate the feature vectors for all layers
    input_feature_vectors = generate_input_feature_vectors(lower_bound, upper_bound, perturbed_image,
                                                           gradient_info_dict)
    relu_feature_vectors_list, output_feature_vectors = generate_relu_output_feature_vectors(simplified_model,
                                                                                             lower_bound,
                                                                                             upper_bound, image,
                                                                                             perturbed_image,
                                                                                             epsilon * epsilon_factor)

    # Initialise the GNN for the given network (which also initialises all the required auxiliary neural networks)
    gnn = GraphNeuralNetwork(simplified_model, image.size(), embedding_vector_size, input_feature_vectors.size()[0],
                             relu_feature_vectors_list[0].size()[0], output_feature_vectors.size()[0],
                             auxiliary_hidden_size, num_update_methods)

    # Follow the GNN framework approach for a specified number of epochs
    for i in range(num_epochs):
        # When the epoch is not the first one, reset the input embedding vectors since the forward input update function
        # only activates when the input embedding vectors are zero
        if i != 0:
            gnn.reset_input_embedding_vectors()

        # Perform a series of forward and backward updates of all the embedding vectors within the GNN
        gnn.update_embedding_vectors(input_feature_vectors, relu_feature_vectors_list, output_feature_vectors,
                                     num_updates)

        # Compute the scores for each image pixel
        pixel_scores = gnn.compute_scores()

        # Update the domain bounds for each pixel based on the pixel scores above
        lower_bound, upper_bound = update_domain_bounds(lower_bound, upper_bound, pixel_scores)

        # Perturb each pixel within the updated domain bounds
        perturbed_image = perturb_image(lower_bound, upper_bound).requires_grad_(True)

        # Perform a PGD attack given the new bounds and perturbation
        successful_attack_flag, gradient_info_dict = gradient_ascent(simplified_model, perturbed_image, lower_bound,
                                                                     upper_bound, pgd_learning_rate, num_iterations)

        # If the attack was successful, the procedure can be terminated and True can be returned, otherwise continue
        if successful_attack_flag:
            return True

        # Otherwise, update all the feature vectors using new information
        input_feature_vectors = generate_input_feature_vectors(lower_bound, upper_bound, perturbed_image,
                                                               gradient_info_dict)
        relu_feature_vectors_list, output_feature_vectors = generate_relu_output_feature_vectors(simplified_model,
                                                                                                 lower_bound,
                                                                                                 upper_bound, image,
                                                                                                 perturbed_image,
                                                                                                 epsilon *
                                                                                                 epsilon_factor)

    # If the limit on the number of epochs was reached and no PGD attack was successful, return False
    return False


def main():
    print(pgd_gnn_attack_properties('base_easy.pkl', 'cifar_base_kw', 1, 0.1, 10, 2, 2, 4, 10, 3, subset=[0]))


if __name__ == '__main__':
    main()
