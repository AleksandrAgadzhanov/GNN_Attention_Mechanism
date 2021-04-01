import torch
import mlogger
import argparse
import time
from exp_utils.model_utils import load_verified_data, match_with_properties
from GNN_framework.helper_functions import match_with_subset, simplify_model, perturb_image, gradient_ascent


def pgd_attack_properties(properties_filename, model_name, epsilon_factor, pgd_learning_rate, num_iterations,
                          num_trials, output_filename, log_filepath=None, subset=None, device='cpu'):
    """
    This function acts as the 2nd baseline to compare the pgd_gnn_attack_property() function against. It initialises a
    specified number of trial random PGD attacks and performs a specified number of iterations of gradient ascent. For
    effective comparison, the product number_of_trials * number_of_iterations_per_trial must match the total number of
    iterations made by pgd_gnn_attack_property(), given by number_of_epochs * number_of_iterations_per_epoch.
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

    # Now attack each property in turn for the specified number of trials. Initialise the counter of properties which
    # were successfully PGD attacked as well as the start time of the experiment. Also initialise the output dictionary
    num_successful_attacks = 0
    start_time = time.time()
    output_dict = {'times': [], 'attack success rates': []}
    for i in range(len(images)):
        # First, simplify the network by adding the final layer and merging the last two layers into one,
        # incorporating the information about the true and test classes into the network
        simplified_model = simplify_model(model, true_labels[i], test_labels[i])
        successful_attack_flag = False

        for trial in range(num_trials):
            # First, perturb the image randomly within the allowed bounds
            lower_bound = torch.add(-epsilons[i] * epsilon_factor, images[i])
            upper_bound = torch.add(epsilons[i] * epsilon_factor, images[i])
            perturbed_image = perturb_image(lower_bound, upper_bound)

            # Now perform a single PGD attack
            successful_attack_flag, _, _ = gradient_ascent(simplified_model, perturbed_image, lower_bound, upper_bound,
                                                           pgd_learning_rate, num_iterations, device=device)

            # If the attack was unsuccessful, increase the counter and break from the loop
            if successful_attack_flag:
                num_successful_attacks += 1
                break

        if log_filepath is not None:
            if successful_attack_flag:
                with mlogger.stdout_to(log_filepath):
                    print('Image ' + str(i + 1) + ' was attacked successfully')
            else:
                with mlogger.stdout_to(log_filepath):
                    print('Image ' + str(i + 1) + ' was NOT attacked successfully')

        # Calculate the attack success rate for the properties in the file provided after all the PGD attacks
        attack_success_rate = 100.0 * num_successful_attacks / len(images)

        # Store the time and current attack success rate in the output dictionary
        output_dict['times'].append(time.time() - start_time)
        output_dict['attack success rates'].append(attack_success_rate)

    # Finally, store the output dictionary in the prescribed location in the current folder
    torch.save(output_dict, 'experiment_results/' + output_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    args = parser.parse_args()
    properties_filename = args.filename + '.pkl'
    log_filepath = 'GNN_framework/' + args.filename + '_log.txt'
    output_filename = args.filename + '_dict.pkl'

    pgd_attack_properties(properties_filename, 'cifar_base_kw', 1.5, 0.1, 100, 101, output_filename,
                          log_filepath=log_filepath, subset=list(range(50)))


if __name__ == '__main__':
    main()
