from matplotlib import pyplot as plt
from GNN_training.train_GNN import generate_gnn_training_parameters


def generate_loss_plot(training_dataset_filename, model_name, gnn_learning_rate, num_epochs, loss_lambda, device='cpu',
                       save_plot=False):
    """
    This function plots the overall loss progression, the progression of the first loss term and the progression of the
    second loss term on the same plot for a specified set of parameters.
    """
    # First, call the training function to obtain a dictionary containing the progression of the two loss terms
    mean_epoch_losses = generate_gnn_training_parameters(training_dataset_filename, model_name, gnn_learning_rate,
                                                         num_epochs, loss_lambda, device=device)

    # Having obtained the above dictionary, construct the lists of first, second loss terms and overall losses
    loss_terms_1 = mean_epoch_losses['loss term 1']
    loss_terms_2_times_lambda = [mean_epoch_losses['loss term 2'][i] * loss_lambda for i in range(len(loss_terms_1))]
    overall_losses = [loss_terms_1[i] + loss_terms_2_times_lambda[i] for i in range(len(loss_terms_1))]

    # Now plot the above loss progressions on the same plot against the epoch number
    plt.plot(range(len(loss_terms_1)), loss_terms_1, color='b', label='Loss term 1')
    plt.plot(range(len(loss_terms_2_times_lambda)), loss_terms_2_times_lambda, color='r', label='Loss term 2 * lambda')
    plt.plot(range(len(overall_losses)), overall_losses, color='k', label='Overall loss')
    plt.legend()
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss terms progression vs epoch number (Learning rate: ' + str(gnn_learning_rate) +
              '; Number of epochs: ' + str(num_epochs) + '; Lambda: ' + str(loss_lambda) + ')')
    plt.xticks(range(0, len(loss_terms_1) + 1, 5), range(0, len(loss_terms_1) + 1, 5))

    # If the save_plot argument is true, save it using a special filename
    if save_plot:
        plt.savefig('../graphs/loss_plot_lr_' + str(gnn_learning_rate) + '_ep_' + str(num_epochs) + '_lam_' +
                    str(loss_lambda) + '.png')

    plt.show()


def main():
    generate_loss_plot('val_SAT_jade_dataset.pkl', 'cifar_base_kw', 0.0001, 25, 2.15, device='cuda')


if __name__ == '__main__':
    main()
