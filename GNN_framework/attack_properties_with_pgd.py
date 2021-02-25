import torch
import mlogger
from exp_utils.model_utils import load_verified_data, match_with_properties
from GNN_framework.GraphNeuralNetwork import GraphNeuralNetwork
from GNN_framework.helper_functions import match_with_subset, simplify_model, perturb_image, gradient_ascent
from GNN_framework.features_generation import generate_input_feature_vectors, generate_relu_output_feature_vectors


def pgd_gnn_attack_properties(properties_filename, model_name, epsilon_factor, pgd_learning_rate, num_iterations,
                              num_attack_epochs, num_trials, num_restarts, gnn_parameters_filename, log_filename=None,
                              subset=None, device='cpu'):
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
    num_successful_attacks = 0  # counter of properties which were successfully PGD attacked
    for i in range(len(images)):
        # First, simplify the network by adding the final layer and merging the last two layers into one, incorporating
        # the information about the true and test classes into the network
        simplified_model = simplify_model(model, true_labels[i], test_labels[i])

        successful_attack_flag = pgd_gnn_attack_property(simplified_model, images[i], epsilons[i], epsilon_factor,
                                                         pgd_learning_rate, num_iterations, num_attack_epochs,
                                                         num_trials, num_restarts, gnn_parameters_filename,
                                                         log_filename, device=device)

        if log_filename is not None:
            if successful_attack_flag:
                with mlogger.stdout_to('GNN_training/' + log_filename):
                    print('Image ' + str(i + 1) + ' was attacked successfully')
            else:
                with mlogger.stdout_to('GNN_training/' + log_filename):
                    print('Image ' + str(i + 1) + ' was NOT attacked successfully')
        else:
            if successful_attack_flag:
                print('Image ' + str(i + 1) + ' was attacked successfully')
            else:
                print('Image ' + str(i + 1) + ' was NOT attacked successfully')

        # If the attack was unsuccessful, increase the counter
        if successful_attack_flag:
            num_successful_attacks += 1

    # Calculate the attack success rate for the properties in the file provided after all the PGD attacks
    attack_success_rate = 100.0 * num_successful_attacks / len(images)

    return attack_success_rate


def pgd_gnn_attack_property(simplified_model, image, epsilon, epsilon_factor, pgd_learning_rate, num_iterations,
                            num_attack_epochs, num_trials, num_restarts, gnn_parameters_filename, log_filename=None,
                            device='cpu'):
    """
    This function performs the PGD attack on the specified property characterised by its image, corresponding simplified
    model and epsilon value by utilising the GNN framework. The first PGD attack is important so a number of restarts is
    specified by one of the arguments. During each restart, the GNN performs a specified number of bound updates and
    after each such update a specified number of trial PGD attacks are performed on the new domain.
    """
    # For a specified number of restarts
    for restart in range(num_restarts):

        # First, perturb the image randomly within the allowed bounds and perform a PGD attack
        lower_bound = torch.add(-epsilon * epsilon_factor, image)
        upper_bound = torch.add(epsilon * epsilon_factor, image)
        perturbed_image = perturb_image(lower_bound, upper_bound)
        successful_attack_flag, perturbed_image, gradient_info_dict = gradient_ascent(simplified_model, perturbed_image,
                                                                                      lower_bound, upper_bound,
                                                                                      pgd_learning_rate, num_iterations,
                                                                                      device=device)

        # If the attack was successful, the procedure can be terminated and True can be returned
        if successful_attack_flag:
            if log_filename is not None:
                with mlogger.stdout_to('GNN_training/' + log_filename):
                    print("Initial PGD attack succeeded")
            else:
                print("Initial PGD attack succeeded")
            return True

        # Otherwise, the GNN framework approach must be followed. First, generate the feature vectors for all layers
        input_feature_vectors = generate_input_feature_vectors(lower_bound, upper_bound, perturbed_image,
                                                               gradient_info_dict)
        relu_feature_vectors_list, output_feature_vectors = generate_relu_output_feature_vectors(simplified_model,
                                                                                                 lower_bound,
                                                                                                 upper_bound,
                                                                                                 perturbed_image)

        # Initialise the GNN for the given network (which also initialises all the required auxiliary neural networks)
        gnn = GraphNeuralNetwork(simplified_model, image.size(), input_feature_vectors.size()[0],
                                 relu_feature_vectors_list[0].size()[0], output_feature_vectors.size()[0])

        # Load the learnt GNN parameters into the GNN
        gnn.load_parameters(gnn_parameters_filename)

        # Follow the GNN framework approach for a specified number of epochs
        for attack_epoch in range(num_attack_epochs):
            # Perform a series of forward and backward updates of all the embedding vectors within the GNN
            gnn.update_embedding_vectors(input_feature_vectors, relu_feature_vectors_list, output_feature_vectors)

            # Update the domain bounds for each pixel based on the pixel scores above
            lower_bound, upper_bound = gnn.compute_updated_bounds(lower_bound, upper_bound)

            # For a specified number of random restarts, perform randomly initialised PGD attacks on the new subdomain
            for trial in range(num_trials):

                # Perturb each pixel within the updated domain bounds
                perturbed_image = perturb_image(lower_bound, upper_bound)

                # Perform a PGD attack given the new bounds and perturbation
                successful_attack_flag, perturbed_image, gradient_info_dict = gradient_ascent(simplified_model,
                                                                                              perturbed_image,
                                                                                              lower_bound, upper_bound,
                                                                                              pgd_learning_rate,
                                                                                              num_iterations,
                                                                                              device=device)

                # If the attack was successful, the procedure can be terminated and True can be returned, otherwise
                # continue
                if successful_attack_flag:
                    if log_filename is not None:
                        with mlogger.stdout_to('GNN_training/' + log_filename):
                            print("PGD attack succeeded during: (Trial " + str(trial + 1) + "; Attack Epoch " +
                                  str(attack_epoch + 1) + "; Restart " + str(restart) + ")")
                    else:
                        print("PGD attack succeeded during: (Trial " + str(trial + 1) + "; Attack Epoch " +
                              str(attack_epoch + 1) + "; Restart " + str(restart) + ")")
                    return True

            # Otherwise, update all the feature vectors using new information
            input_feature_vectors = generate_input_feature_vectors(lower_bound, upper_bound, perturbed_image,
                                                                   gradient_info_dict)

            relu_feature_vectors_list, output_feature_vectors = generate_relu_output_feature_vectors(simplified_model,
                                                                                                     lower_bound,
                                                                                                     upper_bound,
                                                                                                     perturbed_image)

    # If the limit on the number of restarts was reached and no PGD attack was successful, return False
    return False


def main():
    pgd_gnn_attack_properties('val_SAT_jade.pkl', 'cifar_base_kw', 1.0, 0.1, 100, 3, 1, 3, 'gnn_parameters_0.108.pkl',
                              device='cuda')


if __name__ == '__main__':
    main()
