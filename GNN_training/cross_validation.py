import numpy as np
from matplotlib import pyplot as plt
from GNN_framework.attack_properties_with_pgd import pgd_gnn_attack_properties
from GNN_training.train_GNN import generate_gnn_training_parameters


def main():
    loss_lambdas = np.linspace(0, 1, 6)
    training_attack_success_rates = []
    validation_attack_success_rates = []

    for loss_lambda in loss_lambdas:
        generate_gnn_training_parameters('../cifar_exp/training_dataset_subset_50.pkl', 'cifar_base_kw', 0.001, 10,
                                         loss_lambda, '../cifar_exp/temp_parameters_50.pkl')
        print('\nTrained the GNN with lambda = ' + str(loss_lambda))

        training_attack_success_rates.append(pgd_gnn_attack_properties('train_SAT_med.pkl',
                                                                       'cifar_base_kw', 1, 0.1, 100, 5,
                                                                       '../cifar_exp/temp_parameters_50.pkl'))
        print('Performed PGD attacks on the training dataset. Attack success rate is ' +
              str(training_attack_success_rates[-1] + '%'))

        validation_attack_success_rates.append(pgd_gnn_attack_properties('val_SAT_jade.pkl', 'cifar_base_kw', 1, 0.1,
                                                                         100, 5, '../cifar_exp/temp_parameters_50.pkl'))
        print('Performed PGD attacks on the validation dataset. Attack success rate is ' +
              str(validation_attack_success_rates[-1] + '%'))

    plt.plot(loss_lambdas, training_attack_success_rates, color='b', label='Training dataset')
    plt.plot(loss_lambdas, validation_attack_success_rates, color='r', label='Validation dataset')
    plt.xlabel('Loss Lambda')
    plt.ylabel('Attack success rate')
    plt.title('Cross-validation attack success rates')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
