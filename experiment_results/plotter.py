import glob
import torch
from matplotlib import pyplot as plt


def plot_cross_validation_results(directory, space='log'):
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
    if space == 'log':
        plt.semilogx(lambdas, attack_success_rates, color='b')
    elif space == 'lin':
        plt.plot(lambdas, attack_success_rates, color='b')
    plt.xlabel('Lambda')
    plt.ylabel('Attack success rate (%)')
    plt.title('Cross-validation results')
    plt.ylim([0, 100])
    plt.yticks(range(0, 101, 10))
    plt.grid()
    plt.show()


def plot_training_loss(filepath):
    """
    This function generates the same training loss plot as the function in the GNN training folder but it does so by
    using a ready dictionary specified by its respective value of lambda.
    """
    # Load the required dictionary corresponding to the input filename
    dictionary = torch.load(filepath)

    # Having obtained the above dictionary, construct the lists of first, second loss terms and overall losses
    loss_lambda = dictionary['lambda']
    loss_terms_1 = dictionary['loss term 1']
    loss_terms_2_times_lambda = [dictionary['loss term 2'][i] * loss_lambda for i in
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
    plt.xticks(range(0, len(loss_terms_1) + 1, 10), range(0, len(loss_terms_1) + 1, 10))
    plt.show()


def plot_attack_success_rates(filepath_gnn_dict, filepath_baseline_dict, title):
    """
    This function produces two plots of attack success rate versus time on the same figure: one using the dictionary
    corresponding to the GNN attacks and one using the dictionary corresponding to the baseline attacks.
    """
    # First, load the required dictionaries from each specified filepath
    gnn_dict = torch.load(filepath_gnn_dict)
    baseline_dict = torch.load(filepath_baseline_dict)

    # Extract the times and corresponding attack success rates from the dictionaries
    gnn_times = gnn_dict['times']
    baseline_times = baseline_dict['times']
    gnn_attack_success_rates = gnn_dict['attack success rates']
    baseline_attack_success_rates = baseline_dict['attack success rates']

    # Convert the times from seconds to hours for better visualisation
    gnn_times = [1.0 * time / 3600 for time in gnn_times]
    baseline_times = [1.0 * time / 3600 for time in baseline_times]

    # Finally, produce an annotated plot
    plt.plot(gnn_times, gnn_attack_success_rates, color='b', label='GNN attacks')
    plt.plot(baseline_times, baseline_attack_success_rates, color='r', label='Baseline attacks')
    plt.title(title)
    plt.xlabel('Time (hrs)')
    plt.ylabel('Attack success rate (%)')
    plt.grid()
    plt.legend()
    plt.ylim([0, 100])
    plt.yticks(range(0, 101, 10), range(0, 101, 10))
    plt.text(gnn_times[-1], gnn_attack_success_rates[-1], str(round(gnn_attack_success_rates[-1], 1)))
    plt.text(baseline_times[-1], baseline_attack_success_rates[-1], str(round(baseline_attack_success_rates[-1], 1)))
    plt.show()


def main():
    # plot_training_loss('../experiment_results/GNN_2_zooms/training_dict.pkl')
    plot_cross_validation_results(directory='../experiment_results/GNN_3_zooms/cross_validation_2nd_iteration/',
                                  space='lin')
    # plot_attack_success_rates('GNN_2_zooms/test_attacks_gnn_dict.pkl',
    #                           'GNN_2_zooms/test_attacks_baseline_dict.pkl',
    #                           'Comparison of the GNN and baseline attacks (test dataset)')


if __name__ == '__main__':
    main()
