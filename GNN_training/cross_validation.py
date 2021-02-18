import mlogger
import numpy as np
from GNN_framework.attack_properties_with_pgd import pgd_gnn_attack_properties
from GNN_training.train_GNN import generate_gnn_training_parameters


def cross_validate_gnn(loss_lambdas, training_dataset_filename, validation_dataset_filename, combined_dataset_filename,
                       model_name, gnn_learning_rate, epsilon_factor, num_training_epochs, pgd_learning_rate,
                       num_iterations, num_attack_epochs, final_parameters_filename):
    """
    This function performs the cross-validation procedure in order to determine the best value of the regularization
    parameter lambda for learning the parameters of the Graph Neural Network. After finding this best value, the
    parameters are trained using it on the combined training and validation dataset and saved to the specified filename.
    """
    # Initialize the list where the attack success rates on the validation dataset will be stored
    validation_attack_success_rates = []

    # Initialize the variables storing the highest attack success rate on the validation dataset and the corresponding
    # value of lambda
    best_attack_success_rate = 0
    best_lambda = loss_lambdas[0]

    # Try each value of lambda in the list
    for loss_lambda in loss_lambdas:
        # Train the GNN using the current value of lambda and output the learnt parameters in the temporary file
        generate_gnn_training_parameters(training_dataset_filename, model_name, gnn_learning_rate, num_training_epochs,
                                         loss_lambda, 'cifar_exp/temp_gnn_parameters.pkl')
        with mlogger.stdout_to('cross_validation_log.txt'):
            print('\nTrained the GNN with lambda = ' + str(loss_lambda))

        # Let the GNN perform PGD attacks on the validation dataset. Store the resulting attack success rate in the list
        validation_attack_success_rate = pgd_gnn_attack_properties(validation_dataset_filename, model_name,
                                                                   epsilon_factor, pgd_learning_rate, num_iterations,
                                                                   num_attack_epochs,
                                                                   'cifar_exp/temp_gnn_parameters.pkl')
        validation_attack_success_rates.append(validation_attack_success_rate)
        with mlogger.stdout_to('cross_validation_log.txt'):
            print('Performed PGD attacks on the validation dataset. Attack success rate = ' +
                  str(validation_attack_success_rates[-1] + '%'))

        # If the current validation attack success rate is higher than the previous best one, update the variables
        # storing the best attack success rate and lambda
        if validation_attack_success_rate > best_attack_success_rate:
            best_attack_success_rate = validation_attack_success_rate
            best_lambda = loss_lambda

    # # Now that the best lambda parameter is obtained, train the GNN on the combined training and validation dataset and
    # # store the final set of parameters in a special file
    # generate_gnn_training_parameters(combined_dataset_filename, model_name, gnn_learning_rate, num_training_epochs,
    #                                  best_lambda, final_parameters_filename)


def main():
    lambdas = np.logspace(0.01, 10, 4)
    cross_validate_gnn(lambdas, 'train_SAT_med_dataset.pkl', 'val_SAT_jade.pkl', '', 'cifar_base_kw', 0.01, 1, 10, 0.01,
                       2000, 10, '')


if __name__ == '__main__':
    main()
