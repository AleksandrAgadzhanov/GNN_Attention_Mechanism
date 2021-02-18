import torch
import mlogger
from exp_utils.model_utils import load_trained_model
from GNN_framework.GraphNeuralNetwork import GraphNeuralNetwork
from GNN_framework.helper_functions import simplify_model
from GNN_training.helper_functions import compute_loss


def generate_gnn_training_parameters(training_dataset_filename, model_name, gnn_learning_rate, num_epochs, loss_lambda,
                                     output_filename, device='cpu'):
    """
    This function performs training of a Graph Neural Network by utilising supervised learning. After the parameters of
    the Graph Neural Network are learned, they are stored in a desired file.
    """
    # First, load the training dataset which is a list of feature dictionaries from the specified filename. Also load
    # the model
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

    # Follow the training algorithm for a specified number of epochs
    for epoch in range(num_epochs):
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
                # TODO
                new_lower_bound = new_lower_bound.cuda()
                print(new_lower_bound)
                new_upper_bound = new_upper_bound.cuda()

            # Compute the loss by making a call to the special function
            loss = compute_loss(new_lower_bound, new_upper_bound, feature_dict['successful attack'], loss_lambda,
                                device=device)
            if device == 'cuda' and torch.cuda.is_available():
                loss = loss.cuda()

            # Make the optimizer step in a usual manner
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print a message to the terminal at the end of each epoch
        with mlogger.stdout_to('GNN_training/cross_validation_log.txt'):
            print("Epoch №" + str(epoch + 1) + " complete")

    # Finally, after training is finished, construct a list of all the state dictionaries of the auxiliary neural
    # networks of the GNN
    gnn_state_dicts_list = []
    gnn_neural_networks = [gnn.forward_input_update_nn,
                           gnn.forward_relu_update_nn,
                           gnn.forward_output_update_nn,
                           gnn.backward_relu_update_nn,
                           gnn.backward_input_update_nn,
                           gnn.bounds_update_nn]
    for gnn_neural_network in gnn_neural_networks:
        gnn_state_dicts_list.append(gnn_neural_network.state_dict())
    torch.save(gnn_state_dicts_list, 'cifar_exp/' + output_filename)


def main():
    pass


if __name__ == '__main__':
    main()
