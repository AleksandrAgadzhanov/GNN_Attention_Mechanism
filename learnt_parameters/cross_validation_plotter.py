import glob
import torch
from matplotlib import pyplot as plt


def plot_cross_validation_results(directory=""):
    """
    This function plots a graph of the attack success rate against the cross-validation hyper-parameter lambda on the
    log-linear scale. It uses all the .pkl cross validation dictionaries it finds in the current directory for plotting.
    """
    # First, extract all the cross validation dictionaries into a list
    filenames_list = glob.glob(directory + 'cross_validation_dict_*')

    # Sort the filenames by the value of lambda indicated in their name
    sorted_filenames_list = sorted(filenames_list, key=lambda name: float(name[(len(directory) + 22):-4]))

    # Now load the cross validation dictionaries
    cross_validation_dicts = [torch.load(filename) for filename in sorted_filenames_list]

    # Extract the lambda values from the dictionaries
    lambdas = [cross_validation_dict['lambda'] for cross_validation_dict in cross_validation_dicts]

    # Extract the attack success rates from the dictionaries
    attack_success_rates = [cross_validation_dict['attack success rate'] for cross_validation_dict in
                            cross_validation_dicts]

    # Finally, plot the attack success rates vs lambda values on the log-linear scale
    plt.semilogx(lambdas, attack_success_rates, color='b')
    plt.xlabel('Lambda')
    plt.ylabel('Attack success rate (%)')
    plt.title('Cross-validation results')
    plt.ylim([0, 100])
    plt.yticks(range(0, 101, 10))
    plt.grid()
    plt.show()


def plot_training_loss(loss_lambda, directory=""):
    """
    This function generates the same training loss plot as the function in the GNN training folder but it does so by
    using a ready cross validation dictionary specified by its respective value of lambda.
    """
    # Load the required cross validation dictionary corresponding to the input value of lambda
    cross_validation_dict = torch.load(directory + 'cross_validation_dict_' + str(loss_lambda) + '.pkl')

    # Having obtained the above dictionary, construct the lists of first, second loss terms and overall losses
    loss_terms_1 = cross_validation_dict['loss term 1']
    loss_terms_2_times_lambda = [cross_validation_dict['loss term 2'][i] * loss_lambda for i in
                                 range(len(loss_terms_1))]
    overall_losses = [loss_terms_1[i] + loss_terms_2_times_lambda[i] for i in range(len(loss_terms_1))]

    # Now plot the above loss progressions on the same plot against the epoch number
    plt.plot(range(len(loss_terms_1)), loss_terms_1, color='b', label='Loss term 1')
    plt.plot(range(len(loss_terms_2_times_lambda)), loss_terms_2_times_lambda, color='r', label='Loss term 2 * lambda')
    plt.plot(range(len(overall_losses)), overall_losses, color='k', label='Overall loss')
    plt.legend()
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss terms progression vs epoch number (Lambda: ' + str(loss_lambda) + ')')
    plt.xticks(range(0, len(loss_terms_1) + 1, 5), range(0, len(loss_terms_1) + 1, 5))
    plt.show()


def main():
    plot_cross_validation_results(directory='../learnt_parameters/GNN_1_zoom/')


if __name__ == '__main__':
    main()
