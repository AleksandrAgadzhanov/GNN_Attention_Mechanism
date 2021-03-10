import torch
import mlogger
from exp_utils.model_utils import load_trained_model
from GNN_framework.GraphNeuralNetwork import GraphNeuralNetwork
from GNN_framework.helper_functions import simplify_model
from GNN_training.helper_functions import compute_loss
import glob


def generate_gnn_training_parameters(training_dataset_filename, model_name, gnn_learning_rate, num_epochs, loss_lambda,
                                     output_filename=None, log_filename=None, device='cpu'):
    """
    This function performs training of a Graph Neural Network by utilising supervised learning. After the parameters of
    the Graph Neural Network are learned, they are stored in a desired file.
    """
    # First, load the training dataset which is a list of feature dictionaries from the specified filename. Also load
    # the model. If the combined training dataset is specified, construct it from its constituent parts
    if training_dataset_filename == 'train_SAT_jade_combined_dataset.pkl':
        # First, extract all the parts of the training dataset into a list
        filenames_list = glob.glob('cifar_exp/train_SAT_jade_combined_dataset_*')

        # Finally, construct the list of dictionaries fromm the parts of the overall training dataset
        list_of_feature_dicts = []
        for filename in filenames_list:
            list_of_feature_dicts += torch.load(filename)

    # If the combined training and validation dataset is specified, construct it from all the relevant parts
    elif training_dataset_filename == 'train_val_SAT_jade_combined_dataset.pkl':
        # First, extract all the parts of the training dataset into a list
        training_filenames_list = glob.glob('cifar_exp/train_SAT_jade_combined_dataset_*')

        # Construct the list of dictionaries fromm the parts of the overall training dataset
        list_of_feature_dicts = []
        for filename in training_filenames_list:
            list_of_feature_dicts += torch.load(filename)

        # Finally, add the validation dataset information to the overall list
        list_of_feature_dicts += torch.load('cifar_exp/val_SAT_jade_dataset.pkl')

    # Otherwise, load the training dataset all at once
    else:
        list_of_feature_dicts = torch.load('cifar_exp/' + training_dataset_filename)

    model = load_trained_model(model_name)

    # Create the temporary variables which will only be used to initialise the GNN structure. Then create an instance of
    # the GraphNeuralNetwork object using these variables
    temp_simplified_model = simplify_model(model, 0, 1)
    temp_input_size = list_of_feature_dicts[0]['successful attack'].size()
    temp_input_feature_size = list_of_feature_dicts[0]['input'].size()[0]
    temp_relu_feature_size = list_of_feature_dicts[0]['hidden'][0].size()[0]
    temp_output_feature_size = list_of_feature_dicts[0]['output'].size()[0]
    gnn = GraphNeuralNetwork(temp_simplified_model, temp_input_size, temp_input_feature_size, temp_relu_feature_size,
                             temp_output_feature_size, training_mode=True, device=device)

    if device == 'cuda' and torch.cuda.is_available():
        for dict_idx in range(len(list_of_feature_dicts)):
            list_of_feature_dicts[dict_idx]['input'] = list_of_feature_dicts[dict_idx]['input'].cuda()
            list_of_feature_dicts[dict_idx]['hidden'] = [
                tensor.cuda() for tensor in list_of_feature_dicts[dict_idx]['hidden']]
            list_of_feature_dicts[dict_idx]['output'] = list_of_feature_dicts[dict_idx]['output'].cuda()
            list_of_feature_dicts[dict_idx]['successful attack'] = list_of_feature_dicts[dict_idx][
                'successful attack'].cuda()

    # Initialise the optimizer on the parameters of the GNN
    optimizer = torch.optim.Adam(gnn.parameters(), lr=gnn_learning_rate)

    # Initialize the dictionary to store both epoch loss terms progression in
    mean_epoch_losses = {'loss term 1': [], 'loss term 2': []}

    # Follow the training algorithm for a specified number of epochs
    for epoch in range(num_epochs):
        # Initialize the variable which will accumulate the losses related to both loss terms over each epoch
        epoch_loss_term_1 = 0
        epoch_loss_term_2 = 0

        # For each property appearing in the training dataset
        for property_index in range(len(list_of_feature_dicts)):
            feature_dict = list_of_feature_dicts[property_index]

            # Update the last layer of the GNN according to the currently considered true and test labels
            gnn.reconnect_last_layer(model_name, feature_dict['true label'], feature_dict['test label'])

            # Perform a series of forward and backward updates of all the embedding vectors within the GNN
            gnn.update_embedding_vectors(feature_dict['input'], feature_dict['hidden'], feature_dict['output'])

            # Update the domain bounds for each pixel based on the GNN outputs
            old_lower_bound = feature_dict['input'][0, :].reshape(temp_input_size)
            old_upper_bound = feature_dict['input'][1, :].reshape(temp_input_size)
            if device == 'cuda' and torch.cuda.is_available():
                old_lower_bound = old_lower_bound.cuda()
                old_upper_bound = old_upper_bound.cuda()
            new_lower_bound, new_upper_bound = gnn.compute_updated_bounds(old_lower_bound, old_upper_bound)
            if device == 'cuda' and torch.cuda.is_available():
                new_lower_bound = new_lower_bound.cuda()
                new_upper_bound = new_upper_bound.cuda()

            # Compute the loss by making a call to the special function and add it to the accumulator variable
            loss, loss_term_1, loss_term_2 = compute_loss(old_lower_bound, old_upper_bound, new_lower_bound,
                                                          new_upper_bound, feature_dict['successful attack'],
                                                          loss_lambda, device=device)
            epoch_loss_term_1 += loss_term_1.item()
            epoch_loss_term_2 += loss_term_2.item()
            if device == 'cuda' and torch.cuda.is_available():
                loss = loss.cuda()

            # Make the optimizer step in a usual manner
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Append the mean epoch losses during the current epoch to the list
        mean_epoch_losses['loss term 1'].append(epoch_loss_term_1 / len(list_of_feature_dicts))
        mean_epoch_losses['loss term 2'].append(epoch_loss_term_2 / len(list_of_feature_dicts))

        # Print a message to the terminal at the end of each epoch
        if log_filename is not None:
            with mlogger.stdout_to('GNN_training/' + log_filename):
                print("Epoch " + str(epoch + 1) + " complete")
        else:
            print("Epoch " + str(epoch + 1) + " complete")

    if output_filename is not None:
        # Finally, after training is finished, if an output filename was specified, construct a list of all the state
        # dictionaries of the auxiliary neural networks of the GNN and save it
        gnn_state_dicts_list = []

        gnn_neural_networks = [gnn.forward_input_update_nn,
                               gnn.forward_relu_update_nn,
                               gnn.forward_output_update_nn,
                               gnn.backward_relu_update_nn,
                               gnn.backward_input_update_nn,
                               gnn.bounds_update_nn]
        for gnn_neural_network in gnn_neural_networks:
            gnn_state_dicts_list.append(gnn_neural_network.state_dict())

        torch.save(gnn_state_dicts_list, 'learnt_parameters/' + output_filename)

    return mean_epoch_losses


def main():
    mean_epoch_losses = generate_gnn_training_parameters('train_val_SAT_jade_combined_dataset.pkl', 'cifar_base_kw',
                                                         0.0001, 50, 0.066538,
                                                         output_filename='gnn_parameters_1_zoom.pkl',
                                                         log_filename='training_log.txt', device='cuda')
    torch.save(mean_epoch_losses, 'learnt_parameters/training_loss_1_zoom_part_2.pkl')


if __name__ == '__main__':
    main()
