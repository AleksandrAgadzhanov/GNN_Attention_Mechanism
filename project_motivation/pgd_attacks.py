from exp_utils.model_utils import load_verified_data
import torch
import copy
import time
import pandas as pd
import numpy as np
import mlogger


def pgd_attack_properties_old(properties_filename, model_name, attack_method_trials, epsilon_percent, pgd_learning_rate,
                              num_epochs, num_restarts, log_filepath=None, subset=None):
    # Load all the required data for the images which were correctly verified by the model
    x_exact, y_true, image_indices, model = load_verified_data(model_name)

    # Load the properties DataFrame, leave only verified properties
    properties_filepath = 'cifar_exp/' + properties_filename
    properties_dataframe = pd.read_pickle(properties_filepath)

    # If the properties dataset is for testing, leave only the correctly verified properties
    if properties_filename == 'base_easy.pkl' or properties_filename == 'base_med.pkl' or \
            properties_filename == 'base_hard.pkl':
        properties_dataframe = properties_dataframe[(properties_dataframe['BSAT_KWOld'] == 'False') |
                                                    (properties_dataframe['BSAT_KW'] == 'False') |
                                                    (properties_dataframe['BSAT_gnnkwT'] == 'False') |
                                                    (properties_dataframe['GSAT'] == 'False') |
                                                    (properties_dataframe['BSAT_gnnkwTO'] == 'False')]

    # Sort the properties DataFrame by the Idx column for the purpose of easier debugging
    properties_dataframe = properties_dataframe.sort_values(by=['Idx'], ascending=True)

    # Drop all the elements of the x_exact, y_true and image_indices which do not appear in the properties file
    properties_image_indices = list(properties_dataframe['Idx'])
    array_length = len(image_indices)
    for i in range(array_length - 1, -1, -1):  # counter starts at the end due to the nature of the pop() function
        image_index = image_indices[i]
        if image_index not in properties_image_indices:
            x_exact.pop(i)
            y_true.pop(i)
            image_indices.pop(i)

    # Create the list of classes the properties were verified against and the list of epsilons
    y_test = []
    epsilons = []
    for i in range(len(x_exact)):
        y_test.append(properties_dataframe[properties_dataframe['Idx'] == image_indices[i]].iloc[0]['prop'])
        epsilons.append(properties_dataframe[properties_dataframe['Idx'] == image_indices[i]].iloc[0]['Eps'])

    # If the subset of indices was specified for the purpose of reducing the time complexity, drop the elements of
    # x_exact, y_true, y_test and epsilons not indicated in the subset indices
    if subset is not None:
        for i in range(len(image_indices) - 1, -1, -1):
            if i not in subset:
                x_exact.pop(i)
                y_true.pop(i)
                y_test.pop(i)
                epsilons.pop(i)

    # Based on the attack method specified, call the corresponding helper function
    if attack_method_trials[0] == 'random':
        num_trials = attack_method_trials[1]
        attack_success_rate = pgd_attack_properties_random(model, x_exact, y_true, y_test, epsilons, epsilon_percent,
                                                           pgd_learning_rate, num_epochs, num_trials)
        return attack_success_rate
    elif attack_method_trials[0] == 'branch random':
        num_branches = attack_method_trials[1]
        attack_success_rate = pgd_attack_properties_branch_random(model, x_exact, y_true, y_test, epsilons,
                                                                  epsilon_percent, pgd_learning_rate, num_epochs,
                                                                  num_branches)
        return attack_success_rate
    elif attack_method_trials[0] == 'branch heuristic':
        num_branches = attack_method_trials[1]
        output_dict = pgd_attack_properties_branch_heuristic(model, x_exact, y_true, y_test, epsilons,
                                                             epsilon_percent, pgd_learning_rate, num_epochs,
                                                             num_restarts, num_branches, log_filepath=log_filepath)
        return output_dict
    else:
        raise IOError("Please enter a valid attack method (\'random\', \'branch random\' or \'branch heuristic\')")


def logit_difference_loss(logits, test_class, true_class):
    true_class_logit = logits[true_class]
    test_class_logit = logits[test_class]
    return test_class_logit - true_class_logit


def pgd_attack_properties_random(model, x_exact, y_true, y_test, epsilons, epsilon_percent, pgd_learning_rate,
                                 num_epochs, num_trials):
    # Initialise the counter of properties that have been still verified after the PGD attacks
    successfully_attacked_properties = 0

    # Attack each property at a time
    for i in range(len(x_exact)):

        lower_bound = torch.add(-epsilons[i] * (epsilon_percent / 100.0), x_exact[i])
        upper_bound = torch.add(epsilons[i] * (epsilon_percent / 100.0), x_exact[i])
        successful_attack_flag = False

        # Try perturbing the input randomly as many times as indicated by num_trials
        for trial in range(num_trials):
            # Attack the property using gradient ascent
            perturbation = torch.add(-epsilons[i] * (epsilon_percent / 100.0),
                                     2 * (epsilons[i] * (epsilon_percent / 100.0)) * torch.rand(x_exact[i].size()))
            x_perturbed = torch.add(x_exact[i], perturbation).clone().detach().requires_grad_(True)

            successful_attack_flag = pgd_gradient_ascent(model, x_perturbed, lower_bound, upper_bound, y_true[i],
                                                         y_test[i], pgd_learning_rate, num_epochs)

            # If the flag was set to True, there is no need to continue looping through the trials loop
            if successful_attack_flag:
                break

        # If the property withstood all the trial PGD attacks, it is still verified
        if successful_attack_flag:
            successfully_attacked_properties += 1

        print('Image ' + str(i + 1) + ' was processed')

    # Compute the verification accuracy in response to PGD attacks
    attack_success_rate = successfully_attacked_properties / len(x_exact) * 100.0

    return attack_success_rate


def pgd_attack_properties_branch_random(model, x_exact, y_true, y_test, epsilons, epsilon_percent, pgd_learning_rate,
                                        num_epochs, num_branches):
    # Initialise the counter of properties that have been still verified after the PGD attacks
    successfully_attacked_properties = 0

    # Attack each property at a time
    for i in range(len(x_exact)):

        # FIRST PGD ATTACK

        # The first attack is the same as in the randomly initialised PGD attack case
        lower_bound = torch.add(-epsilons[i] * (epsilon_percent / 100.0), x_exact[i])
        upper_bound = torch.add(epsilons[i] * (epsilon_percent / 100.0), x_exact[i])

        perturbation = torch.add(-epsilons[i] * (epsilon_percent / 100.0),
                                 2 * (epsilons[i] * (epsilon_percent / 100.0)) * torch.rand(x_exact[i].size()))
        x_perturbed = torch.add(x_exact[i], perturbation).clone().detach().requires_grad_(True)

        successful_attack_flag = pgd_gradient_ascent(model, x_perturbed, lower_bound, upper_bound, y_true[i],
                                                     y_test[i], pgd_learning_rate, num_epochs)

        # If the flag was set to True after the first attack, move on to the next image
        if successful_attack_flag:
            continue

        # SUBSEQUENT PGD ATTACKS

        # Information about where splits have already been made needs to be stored. Let this information be represented
        # by tensors of the same size as the image where each -1 corresponds to the pixel which can only be perturbed by
        # [-eps, 0], 1 - which can be perturbed by [0, +eps] and 0 - which can be perturbed by [-eps, +eps]

        # Initialise this storage as a list of tensors
        domain_info = [torch.zeros(x_exact[i].size())]  # the only domain is the non-split original domain

        # If the first randomly initialised PGD attack was unsuccessful, splitting procedure will be started: a selected
        # pixel will be split and two subproblems will be generated: one where this pixel's value can be perturbed by
        # [-eps, 0] and one where it can be perturbed by [0, +eps]
        for num_branch in range(num_branches):

            # Choose the domain and the dimension to split on. Images have size ([1, 3, 32, 32]) so it is the
            # last three dimensions that have to be chosen

            # Choose the above randomly
            domain_index = np.random.choice(range(len(domain_info)))
            available_dim_indices = [[matrix_idx, row_idx, value_idx] for matrix_idx in
                                     range(len(domain_info[domain_index][0])) for row_idx in
                                     range(len(domain_info[domain_index][0][matrix_idx])) for value_idx in
                                     range(len(domain_info[domain_index][0][matrix_idx][row_idx]))
                                     if domain_info[domain_index][0][matrix_idx][row_idx][value_idx] == 0]

            dim_indices_idx = np.random.choice(list(range(len(available_dim_indices))))
            dim_indices = available_dim_indices[dim_indices_idx]

            # Create two new subdomains where in one the chosen pixel can be perturbed by [-eps, 0] and in another
            # it can be perturbed by [0, +eps] and append their information tensors to the list
            subdomain_1_info = domain_info[domain_index].clone().detach()
            subdomain_2_info = domain_info[domain_index].clone().detach()
            subdomain_1_info[0][dim_indices[0]][dim_indices[1]][dim_indices[2]] = -1
            subdomain_2_info[0][dim_indices[0]][dim_indices[1]][dim_indices[2]] = 1

            # Remove the information tensor of the original domain on which splitting was performed from the list
            # and append the new information tensors to it
            domain_info.pop(domain_index)
            domain_info.append(subdomain_1_info)
            domain_info.append(subdomain_2_info)

            # Perturb all the pixels of both subdomains and get the corresponding bounds for clipping values
            x_perturbed_1 = perturb_image_special(x_exact[i], subdomain_1_info,
                                                  epsilons[i] * (epsilon_percent / 100.0)).requires_grad_(True)
            lower_bound_1, upper_bound_1 = get_bounds_special(x_exact[i], subdomain_1_info,
                                                              epsilons[i] * (epsilon_percent / 100.0))

            # Attack both subdomains using gradient ascent
            successful_attack_flag = pgd_gradient_ascent(model, x_perturbed_1, lower_bound_1, upper_bound_1,
                                                         y_true[i], y_test[i], pgd_learning_rate, num_epochs)

            # If the attack is successful, move on to the next image by breaking out of the branching loop
            if successful_attack_flag:
                break

            x_perturbed_2 = perturb_image_special(x_exact[i], subdomain_2_info,
                                                  epsilons[i] * (epsilon_percent / 100.0)).requires_grad_(True)
            lower_bound_2, upper_bound_2 = get_bounds_special(x_exact[i], subdomain_2_info,
                                                              epsilons[i] * (epsilon_percent / 100.0))

            successful_attack_flag = pgd_gradient_ascent(model, x_perturbed_2, lower_bound_2, upper_bound_2,
                                                         y_true[i], y_test[i], pgd_learning_rate, num_epochs)

            if successful_attack_flag:
                break

        # If the property withstood all the PGD attacks, it is still verified
        if successful_attack_flag:
            successfully_attacked_properties += 1

    # Compute the verification accuracy in response to PGD attacks
    attack_success_rate = successfully_attacked_properties / len(x_exact) * 100.0

    return attack_success_rate


def pgd_attack_properties_branch_heuristic(model, x_exact, y_true, y_test, epsilons, epsilon_percent, pgd_learning_rate,
                                           num_epochs, num_branches, num_restarts, log_filepath=None):
    # Initialise the counter of properties that have been still verified after the PGD attacks
    successfully_attacked_properties = 0

    # Start the timer and initialise the output dictionary
    start_time = time.time()
    output_dict = {'times': [], 'attack success rates': []}

    # Attack each property at a time
    for i in range(len(x_exact)):
        successful_attack_flag = False

        for restart in range(num_restarts + 1):
            if successful_attack_flag:
                break

            # FIRST PGD ATTACK

            # The first attack is the same as in the randomly initialised PGD attack case
            lower_bound = torch.add(-epsilons[i] * (epsilon_percent / 100.0), x_exact[i])
            upper_bound = torch.add(epsilons[i] * (epsilon_percent / 100.0), x_exact[i])
            heuristic = None

            perturbation = torch.add(-epsilons[i] * (epsilon_percent / 100.0),
                                     2 * (epsilons[i] * (epsilon_percent / 100.0)) * torch.rand(x_exact[i].size()))
            x_perturbed = torch.add(x_exact[i], perturbation).clone().detach().requires_grad_(True)

            output = pgd_gradient_ascent(model, x_perturbed, lower_bound, upper_bound, y_true[i], y_test[i],
                                         pgd_learning_rate, num_epochs, heuristic=True)

            # If only the flag was returned (successful attack case), then set the successful attack flag only
            if type(output) == bool:
                successful_attack_flag = output
            # If both the flag and heuristic were returned, set them both
            else:
                successful_attack_flag = output[0]
                heuristic = output[1]

            # If the flag was set to True after the first attack, move on to the next image while appending time and
            # attack success rate to the output dictionary
            if successful_attack_flag:
                successfully_attacked_properties += 1
                if log_filepath is not None:
                    with mlogger.stdout_to(log_filepath):
                        print('Initial PGD attack succeeded')
                        print('Image ' + str(i + 1) + ' was attacked successfully')
                else:
                    print('Initial PGD attack succeeded')
                    print('Image ' + str(i + 1) + ' was attacked successfully')
                continue

            # SUBSEQUENT PGD ATTACKS

            # Information about where splits have already been made needs to be stored. Let this information be
            # represented by tensors of the same size as the image where each -1 corresponds to the pixel which can
            # only be perturbed by [-eps, 0], 1 - which can be perturbed by [0, +eps] and 0 - which can be perturbed
            # by [-eps, +eps]

            # If the branch type was specified to be heuristic, initialise this storage as a list of lists [information,
            # heuristic]
            domain_info = [[torch.zeros(x_exact[i].size()), heuristic]]

            # If the first randomly initialised PGD attack was unsuccessful, splitting procedure will be started: a
            # selected pixel will be split and two subproblems will be generated: one where this pixel's value can be
            # perturbed by [-eps, 0] and one where it can be perturbed by [0, +eps]
            for num_branch in range(num_branches):

                # Choose the domain and the dimension to split on. Images have size ([1, 3, 32, 32]) so it is the
                # last three dimensions that have to be chosen

                # If the branch type if heuristic, choose the above based on which subdomain contains the largest
                # magnitude of the gradient with respect to a particular pixel
                domain_index = 0
                dim_indices = [0, 0, 0]
                max_gradient_magnitude = 0  # magnitudes are dealt with so 0 is the absolute minimum

                for domain_idx in range(len(domain_info)):
                    for matrix_idx in range(len(domain_info[domain_idx][1][0])):
                        for row_idx in range(len(domain_info[domain_idx][1][0][matrix_idx])):
                            for value_idx in range(len(domain_info[domain_idx][1][0][matrix_idx][row_idx])):
                                gradient_magnitude = domain_info[domain_idx][1][0][matrix_idx][row_idx][value_idx]
                                info_value = domain_info[domain_idx][0][0][matrix_idx][row_idx][value_idx]
                                # If the gradient is larger than the current maximum and splitting on the corresponding
                                # pixel hasn't been done yet, update the variables
                                if gradient_magnitude > max_gradient_magnitude and info_value == 0:
                                    max_gradient_magnitude = gradient_magnitude
                                    domain_index = domain_idx
                                    dim_indices = [matrix_idx, row_idx, value_idx]

                # Same as in the branch random case but now lists are involved
                subdomain_1_list = copy.deepcopy(domain_info[domain_index])
                subdomain_2_list = copy.deepcopy(domain_info[domain_index])
                subdomain_1_list[0][0][dim_indices[0]][dim_indices[1]][dim_indices[2]] = -1
                subdomain_2_list[0][0][dim_indices[0]][dim_indices[1]][dim_indices[2]] = 1

                # Same as in the branch random case but now subdomain information tensors should be set explicitly
                domain_info.pop(domain_index)
                domain_info.append(subdomain_1_list)
                domain_info.append(subdomain_2_list)
                subdomain_1_info = subdomain_1_list[0]
                subdomain_2_info = subdomain_2_list[0]

                # Perturb all the pixels of both subdomains and get the corresponding bounds for clipping values
                x_perturbed_1 = perturb_image_special(x_exact[i], subdomain_1_info,
                                                      epsilons[i] * (epsilon_percent / 100.0)).requires_grad_(True)
                lower_bound_1, upper_bound_1 = get_bounds_special(x_exact[i], subdomain_1_info,
                                                                  epsilons[i] * (epsilon_percent / 100.0))

                # Attack both subdomains using gradient ascent
                # If the specified branch type is heuristic, heuristic parameter is specified to pgd_gradient_ascent
                output_1 = pgd_gradient_ascent(model, x_perturbed_1, lower_bound_1, upper_bound_1, y_true[i], y_test[i],
                                               pgd_learning_rate, num_epochs, heuristic=True)

                # If only the flag was returned (successful attack case), then set the successful attack flag only
                if type(output_1) == bool:
                    successful_attack_flag = output_1
                # If both the flag and heuristic were returned, set them both and update the heuristic information of
                # the subdomain at index -2
                else:
                    successful_attack_flag = output_1[0]
                    heuristic_1 = output_1[1]
                    domain_info[-2][1] = heuristic_1

                # If the attack is successful, move on to the next image by breaking out of the branching loop while
                # storing the time and attack success rate in the output dictionary
                if successful_attack_flag:
                    successfully_attacked_properties += 1
                    if log_filepath is not None:
                        with mlogger.stdout_to(log_filepath):
                            print('Image ' + str(i + 1) + ' was attacked successfully')
                    else:
                        print('Image ' + str(i + 1) + ' was attacked successfully')
                    break

                x_perturbed_2 = perturb_image_special(x_exact[i], subdomain_2_info,
                                                      epsilons[i] * (epsilon_percent / 100.0)).requires_grad_(True)
                lower_bound_2, upper_bound_2 = get_bounds_special(x_exact[i], subdomain_2_info,
                                                                  epsilons[i] * (epsilon_percent / 100.0))

                output_2 = pgd_gradient_ascent(model, x_perturbed_2, lower_bound_2, upper_bound_2, y_true[i], y_test[i],
                                               pgd_learning_rate, num_epochs, heuristic=True)

                if type(output_2) == bool:
                    successful_attack_flag = output_2
                else:
                    successful_attack_flag = output_2[0]
                    heuristic_2 = output_2[1]
                    domain_info[-1][1] = heuristic_2  # information about heuristic of the last subdomain is updated

                if successful_attack_flag:
                    successfully_attacked_properties += 1
                    if log_filepath is not None:
                        with mlogger.stdout_to(log_filepath):
                            print('Image ' + str(i + 1) + ' was attacked successfully')
                    else:
                        print('Image ' + str(i + 1) + ' was attacked successfully')
                    break

        if not successful_attack_flag:
            if log_filepath is not None:
                with mlogger.stdout_to(log_filepath):
                    print('Image ' + str(i + 1) + ' was NOT attacked successfully')
            else:
                print('Image ' + str(i + 1) + ' was NOT attacked successfully')

        # Calculate the new attack success rate and append it and the time to the output dictionary
        attack_success_rate = 100.0 * successfully_attacked_properties / len(x_exact)
        output_dict['times'].append(time.time() - start_time)
        output_dict['attack success rates'].append(attack_success_rate)

    return output_dict


def pgd_gradient_ascent(model, x_perturbed, lower_bound, upper_bound, y_true, y_test, pgd_learning_rate, num_epochs,
                        heuristic=False):
    successful_attack_flag = False

    optimizer = torch.optim.Adam([x_perturbed], lr=pgd_learning_rate)

    # Perform Gradient Ascent for a specified number of epochs
    for epoch in range(num_epochs):
        logits = model(x_perturbed)[0]
        loss = -logit_difference_loss(logits, y_test, y_true)  # '-' sign since gradient ascent is performed

        # If the difference between the logit of the test class and the logit of the true class is positive,
        # then the PGD attack was successful and gradient ascent can be stopped
        if -loss > 0:
            successful_attack_flag = True
            return successful_attack_flag

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clip the values of the perturbed image so that they are within the allowed perturbation magnitude
        # This operation isn't related to optimisation, hence it is wrapped with torch.no_grad()
        with torch.no_grad():
            x_perturbed[:] = torch.max(torch.min(x_perturbed, upper_bound), lower_bound)

    # If the flag has not been set yet but the perturbation resulted in the model predicting an incorrect class
    # during the last epoch, return the True successful attack flag
    if torch.max(model(x_perturbed)[0], 0)[1].item() == y_test:
        successful_attack_flag = True
        return successful_attack_flag

    # If the PGD attack was unsuccessful and heuristic is needed for the output, return the gradient magnitudes together
    # with the attack flag
    if heuristic is True:
        gradient_magnitudes = torch.abs(x_perturbed.grad)
        return successful_attack_flag, gradient_magnitudes

    return successful_attack_flag


def perturb_image_special(x_exact, information_tensor, epsilon):
    perturbation = torch.zeros(x_exact.size())

    # Perturb all the pixels by the amounts specified in the relevant information tensor
    for matrix_idx in range(len(information_tensor[0])):
        for row_idx in range(len(information_tensor[0][matrix_idx])):
            for value_idx in range(len(information_tensor[0][matrix_idx][row_idx])):
                info_value = information_tensor[0][matrix_idx][row_idx][value_idx]
                if info_value == -1:
                    perturbation[0][matrix_idx][row_idx][value_idx] = np.random.uniform(-epsilon, 0)
                elif info_value == 1:
                    perturbation[0][matrix_idx][row_idx][value_idx] = np.random.uniform(0, epsilon)
                else:
                    perturbation[0][matrix_idx][row_idx][value_idx] = np.random.uniform(-epsilon, epsilon)

    x_perturbed = torch.add(x_exact, perturbation).clone().detach()
    return x_perturbed


def get_bounds_special(x_exact, information_tensor, epsilon):
    # Initialise the bounds to the exact pixel values
    lower_bound = x_exact.clone().detach()
    upper_bound = x_exact.clone().detach()

    # Evaluate the bounds for each pixel based on the relevant information tensor
    for matrix_idx in range(len(information_tensor[0])):
        for row_idx in range(len(information_tensor[0][matrix_idx])):
            for value_idx in range(len(information_tensor[0][matrix_idx][row_idx])):
                info_value = information_tensor[0][matrix_idx][row_idx][value_idx]
                if info_value == -1:
                    lower_bound[0][matrix_idx][row_idx][value_idx] -= epsilon
                    # The upper bound is taken directly from x_exact
                elif info_value == 1:
                    # The lower bound is taken directly from x_exact
                    upper_bound[0][matrix_idx][row_idx][value_idx] += epsilon
                else:
                    lower_bound[0][matrix_idx][row_idx][value_idx] -= epsilon
                    upper_bound[0][matrix_idx][row_idx][value_idx] += epsilon

    return lower_bound, upper_bound


def main():
    output_dict_heuristics = pgd_attack_properties_old('base_easy_SAT_jade.pkl', 'cifar_base_kw',
                                                       ['branch heuristic', 5], 100, 0.1, 100, 9,
                                                       log_filepath='project_motivation/attack_log.txt',
                                                       subset=list(range(100)))
    torch.save(output_dict_heuristics, 'experiment_results/attack_success_rates_heuristics.pkl')


if __name__ == '__main__':
    main()
