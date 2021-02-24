import mlogger
import torch
import time
import argparse
from GNN_framework.attack_properties_with_pgd import pgd_gnn_attack_properties
from GNN_training.train_GNN import generate_gnn_training_parameters


def cross_validate_gnn(loss_lambda, training_dataset_filename, validation_properties_filename, model_name,
                       gnn_learning_rate, num_training_epochs, pgd_learning_rate, num_iterations, num_attack_epochs,
                       log_filename=None, epsilon_factor=1.0, device='cpu'):
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
    epoch_losses = generate_gnn_training_parameters(training_dataset_filename, model_name, gnn_learning_rate,
                                                    num_training_epochs, loss_lambda, parameters_filename, log_filename,
                                                    device=device)

    if log_filename is not None:
        with mlogger.stdout_to('GNN_training/' + log_filename):
            print('Epoch losses progression:\n')
            print(epoch_losses)
    else:
        print('Epoch losses progression:\n')
        print(epoch_losses)

    output_dict = {'epoch losses': epoch_losses}
    torch.save(output_dict, output_dictionary_filepath)

    if log_filename is not None:
        with mlogger.stdout_to('GNN_training/' + log_filename):
            print('\nTrained the GNN with lambda = ' + str(loss_lambda))
            print('Time elapsed since the start: ' + str(time.time() - start_time))
    else:
        print('\nTrained the GNN with lambda = ' + str(loss_lambda))
        print('Time elapsed since the start: ' + str(time.time() - start_time))

    # Let the GNN perform PGD attacks on the validation dataset
    validation_attack_success_rate = pgd_gnn_attack_properties(validation_properties_filename, model_name,
                                                               epsilon_factor, pgd_learning_rate, num_iterations,
                                                               num_attack_epochs, parameters_filename, log_filename,
                                                               device=device)

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
    parser.add_argument('--loss_lambda', type=float)
    args = parser.parse_args()

    log_filename = 'cross_validation_log_' + str(args.loss_lambda) + '.pkl'

    cross_validate_gnn(args.loss_lambda, 'train_SAT_jade_dataset.pkl', 'val_SAT_jade.pkl', 'cifar_base_kw', 0.001, 25,
                       0.01, 2000, 10, log_filename=log_filename, device='cuda')


if __name__ == '__main__':
    main()
