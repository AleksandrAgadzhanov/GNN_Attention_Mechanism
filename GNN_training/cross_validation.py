import mlogger
import torch
import time
import argparse
import glob
from GNN_framework.attack_properties_with_pgd import pgd_gnn_attack_properties
from GNN_training.train_GNN import generate_gnn_training_parameters


def cross_validate_gnn(loss_lambda, validation_properties_filename, model_name, pgd_learning_rate, num_iterations,
                       num_attack_epochs, num_trials, num_initialisations, training_dataset_filename=None,
                       gnn_learning_rate=0.00001, num_training_epochs=30, log_filepath=None, epsilon_factor=1.0,
                       device='cpu'):
    """
    This function performs the cross-validation procedure using a single value of lambda.
    """
    # Start the timer
    start_time = time.time()

    # If the train argument was provided, train the GNN first
    if training_dataset_filename:
        # Initialize the filepath in which the GNN parameters and the training losses for the given value of lambda
        # will be stored
        parameters_filepath = 'experiment_results/gnn_parameters_cross_val_' + str(loss_lambda) + '.pkl'
        training_dict_filepath = 'experiment_results/training_dict_' + str(loss_lambda) + '.pkl'

        # Train the GNN using the current value of lambda and output the learnt parameters in the temporary file
        training_dict = generate_gnn_training_parameters(training_dataset_filename, model_name, gnn_learning_rate,
                                                         num_training_epochs, loss_lambda, parameters_filepath,
                                                         log_filepath=log_filepath, device=device)

        if log_filepath is not None:
            with mlogger.stdout_to(log_filepath):
                print('\nTrained GNN with lambda = ' + str(loss_lambda))
                print('Time elapsed since the start: ' + str(time.time() - start_time))
                print('Epoch losses progression:\n')
                print(training_dict)
                print('\n')
        else:
            print('\nTrained GNN with lambda = ' + str(loss_lambda))
            print('Time elapsed since the start: ' + str(time.time() - start_time))
            print('Epoch losses progression:\n')
            print(training_dict)
            print('\n')

        # Save the training dictionary in the appropriate filepath
        torch.save(training_dict, training_dict_filepath)

    # If the train argument was specified as False, skip the training stage. In this case, the parameters should lie in
    # the different directory
    else:
        parameters_filepath = 'experiment_results/cross_validation_gnn_parameters/gnn_parameters_cross_val_' +\
                              str(loss_lambda) + '.pkl'

    # Let the GNN perform PGD attacks on the validation dataset
    validation_attack_success_rate = pgd_gnn_attack_properties(validation_properties_filename, model_name,
                                                               epsilon_factor, pgd_learning_rate, num_iterations,
                                                               num_attack_epochs, num_trials, num_initialisations,
                                                               parameters_filepath, log_filepath=log_filepath,
                                                               device=device)

    if log_filepath is not None:
        with mlogger.stdout_to(log_filepath):
            print('Performed PGD attacks on the validation dataset. Attack success rate = ' +
                  str(validation_attack_success_rate) + '%')
            print('Time elapsed since the start: ' + str(time.time() - start_time))
    else:
        print('Performed PGD attacks on the validation dataset. Attack success rate = ' +
              str(validation_attack_success_rate) + '%')
        print('Time elapsed since the start: ' + str(time.time() - start_time))

    # Initialise the cross validation dictionary and save it in the experiment results dictionary
    cross_validation_dict = {'lambda': loss_lambda,
                             'attack success rate': validation_attack_success_rate}

    torch.save(cross_validation_dict, 'experiment_results/cross_validation_dict_' + str(loss_lambda) + '.pkl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_lambda', type=float)
    parser.add_argument('--end_lambda', type=float)
    parser.add_argument('--num', type=int)
    args = parser.parse_args()

    log_filepath = 'GNN_training/cross_validation_log_' + str(args.start_lambda) + '_to_' + str(args.end_lambda) +\
                   '.pkl'

    import numpy as np

    loss_lambdas = np.linspace(args.start_lambda, args.end_lambda, num=args.num)
    loss_lambdas = [round(loss_lambda, 3) for loss_lambda in loss_lambdas]

    filepaths_list = glob.glob('experiment_results/cross_validation_gnn_parameters/gnn_*')
    filenames_list = [filepath[51:] for filepath in filepaths_list]

    for loss_lambda in loss_lambdas:
        if ('gnn_parameters_cross_val_' + str(loss_lambda) + '.pkl') in filenames_list:
            cross_validate_gnn(loss_lambda, 'val_SAT_jade.pkl', 'cifar_base_kw', 0.1, 100, 1, 29, 3,
                               log_filepath=log_filepath, device='cuda')
        else:
            cross_validate_gnn(loss_lambda, 'val_SAT_jade.pkl', 'cifar_base_kw', 0.1, 100, 1, 29, 3,
                               training_dataset_filename='train_SAT_jade_combined_dataset.pkl',
                               gnn_learning_rate=0.00001, num_training_epochs=30, log_filepath=log_filepath,
                               device='cuda')


if __name__ == '__main__':
    main()
