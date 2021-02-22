import mlogger
import torch
import time
from GNN_framework.attack_properties_with_pgd import pgd_gnn_attack_properties
from GNN_training.train_GNN import generate_gnn_training_parameters


def cross_validate_gnn(loss_lambda, training_dataset_filename, validation_dataset_filename, model_name,
                       gnn_learning_rate, num_training_epochs, pgd_learning_rate, num_iterations, num_attack_epochs,
                       epsilon_factor=1.0, device='cpu'):
    """
    This function performs the cross-validation procedure using a single value of lambda.
    """
    # Start the timer
    start_time = time.time()

    # Initialize the filename in which the parameters for the given value of lambda will be stored
    output_filename = 'gnn_parameters_' + str(loss_lambda) + '.pkl'

    # Train the GNN using the current value of lambda and output the learnt parameters in the temporary file
    epoch_losses = generate_gnn_training_parameters(training_dataset_filename, model_name, gnn_learning_rate,
                                                    num_training_epochs, loss_lambda, output_filename, device=device)

    with mlogger.stdout_to('GNN_training/cross_validation_log.txt'):
        print('\nTrained the GNN with lambda = ' + str(loss_lambda))
        print('Time elapsed since the start: ' + str(time.time() - start_time))

    # Let the GNN perform PGD attacks on the validation dataset
    validation_attack_success_rate = pgd_gnn_attack_properties(validation_dataset_filename, model_name,
                                                               epsilon_factor, pgd_learning_rate, num_iterations,
                                                               num_attack_epochs, output_filename, device=device)

    with mlogger.stdout_to('GNN_training/cross_validation_log.txt'):
        print('Performed PGD attacks on the validation dataset. Attack success rate = ' +
              str(validation_attack_success_rate) + '%')
        print('Time elapsed since the start: ' + str(time.time() - start_time))

    output_dict = {'lambda': loss_lambda, 'attack success rate': validation_attack_success_rate,
                   'epoch losses': epoch_losses}
    torch.save(output_dict, 'cifar_exp/cross_validation_dict_' + str(loss_lambda) + '.pkl')


def main():
    pass


if __name__ == '__main__':
    main()
