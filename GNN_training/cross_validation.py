import mlogger
import torch
import time
import argparse
from GNN_framework.attack_properties_with_pgd import pgd_gnn_attack_properties
from GNN_training.train_GNN import generate_gnn_training_parameters


def cross_validate_gnn(loss_lambda, training_dataset_filename, validation_properties_filename, model_name,
                       gnn_learning_rate, num_training_epochs, pgd_learning_rate, num_iterations, num_attack_epochs,
                       num_trials, num_restarts, log_filename=None, epsilon_factor=1.0, device='cpu'):
    """
    This function performs the cross-validation procedure using a single value of lambda.
    """
    # Start the timer
    start_time = time.time()

    # Initialize the filename in which the parameters for the given value of lambda and will be stored. Also initialize
    # the filepath where the output dictionary for the given value of lambda will be stored
    parameters_filename = 'gnn_parameters_' + str(loss_lambda) + '.pkl'
    output_dictionary_filepath = 'learnt_parameters/cross_validation_dict_' + str(loss_lambda) + '.pkl'

    # Train the GNN using the current value of lambda and output the learnt parameters in the temporary file
    output_dict = generate_gnn_training_parameters(training_dataset_filename, model_name, gnn_learning_rate,
                                                   num_training_epochs, loss_lambda, parameters_filename, log_filename,
                                                   device=device)

    torch.save(output_dict, output_dictionary_filepath)

    if log_filename is not None:
        with mlogger.stdout_to('GNN_training/' + log_filename):
            print('\nTrained GNN with lambda = ' + str(loss_lambda))
            print('Time elapsed since the start: ' + str(time.time() - start_time))
            print('Epoch losses progression:\n')
            print(output_dict)
            print('\n')
    else:
        print('\nTrained GNN with lambda = ' + str(loss_lambda))
        print('Time elapsed since the start: ' + str(time.time() - start_time))
        print('Epoch losses progression:\n')
        print(output_dict)
        print('\n')

    # Let the GNN perform PGD attacks on the validation dataset
    validation_attack_success_rate = pgd_gnn_attack_properties(validation_properties_filename, model_name,
                                                               epsilon_factor, pgd_learning_rate, num_iterations,
                                                               num_attack_epochs, num_trials, num_restarts,
                                                               parameters_filename, log_filename, device=device)

    if log_filename is not None:
        with mlogger.stdout_to('GNN_training/' + log_filename):
            print('Performed PGD attacks on the validation dataset. Attack success rate = ' +
                  str(validation_attack_success_rate) + '%')
            print('Time elapsed since the start: ' + str(time.time() - start_time))
    else:
        print('Performed PGD attacks on the validation dataset. Attack success rate = ' +
              str(validation_attack_success_rate) + '%')
        print('Time elapsed since the start: ' + str(time.time() - start_time))

    output_dict['lambda'] = loss_lambda
    output_dict['attack success rate'] = validation_attack_success_rate

    torch.save(output_dict, output_dictionary_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_lambda', type=float)
    parser.add_argument('--end_lambda', type=float)
    parser.add_argument('--num', type=int)
    parser.add_argument('--endpoint', type=bool)
    args = parser.parse_args()

    log_filename = 'cross_validation_log_' + str(args.start_lambda) + '_to_' + str(args.end_lambda) + '.pkl'

    import numpy as np
    import math

    loss_lambdas = np.logspace(math.log10(args.start_lambda), math.log10(args.end_lambda), num=args.num,
                               endpoint=args.endpoint)

    for loss_lambda in loss_lambdas:
        cross_validate_gnn(loss_lambda, 'train_SAT_jade_dataset.pkl', 'val_SAT_jade.pkl', 'cifar_base_kw', 0.0001, 30,
                           0.1, 100, 1, 20, 1, log_filename=log_filename, device='cuda')


if __name__ == '__main__':
    main()
