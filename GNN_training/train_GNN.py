import torch
from exp_utils.model_utils import load_trained_model
from GNN_framework.GraphNeuralNetwork import GraphNeuralNetwork
from GNN_framework.helper_functions import simplify_model
from GNN_training.helper_functions import compute_loss


def generate_gnn_training_parameters(training_dataset_filename, model_name, gnn_learning_rate, num_epochs,
                                     output_filename):
    """
    This function performs training of a Graph Neural Network by utilising supervised learning. After the parameters of
    the Graph Neural Network are learned, they are stored in a desired file.
    """
    # First, load the training dataset which is a list of feature dictionaries from the specified filename. Also load
    # the model
    list_of_feature_dicts = torch.load('../GNN_training/' + training_dataset_filename)
    model = load_trained_model(model_name)

    # Create the temporary variables which will only be used to initialise the GNN structure. Then create an instance of
    # the GraphNeuralNetwork object using these variables
    temp_simplified_model = simplify_model(model, 0, 1)
    temp_input_size = list_of_feature_dicts[0]['successful attack'].size()
    temp_input_feature_size = list_of_feature_dicts[0]['input'].size()[0]
    temp_relu_feature_size = list_of_feature_dicts[0]['hidden'][0].size()[0]
    temp_output_feature_size = list_of_feature_dicts[0]['output'].size()[0]
    gnn = GraphNeuralNetwork(temp_simplified_model, temp_input_size, temp_input_feature_size, temp_relu_feature_size,
                             temp_output_feature_size, training_mode=True)

    # Initialise the optimizer on the parameters of the GNN
    optimizer = torch.optim.Adam(gnn.parameters(), lr=gnn_learning_rate)
    losses = []
    # Follow the training algorithm for a specified number of epochs
    for epoch in range(num_epochs):
        # For each property appearing in the training dataset
        for property_index in range(len(list_of_feature_dicts)):
            feature_dict = list_of_feature_dicts[property_index]

            # When the epoch or the subdomain index are not the first one, reset the input embedding vectors since the
            # forward input update function only activates when the input embedding vectors are zero
            if epoch != 0 or property_index != 0:
                gnn.reset_input_embedding_vectors()

            # Perform a series of forward and backward updates of all the embedding vectors within the GNN
            gnn.update_embedding_vectors(feature_dict['input'], feature_dict['hidden'], feature_dict['output'])

            # Update the domain bounds for each pixel based on the GNN outputs
            old_lower_bound = feature_dict['input'][0, :].reshape(temp_input_size)
            old_upper_bound = feature_dict['input'][1, :].reshape(temp_input_size)
            new_lower_bound, new_upper_bound = gnn.compute_updated_bounds(old_lower_bound, old_upper_bound)

            # Compute the loss by making a call to the special function
            loss = compute_loss(new_lower_bound, new_upper_bound, feature_dict['successful attack'])
            losses.append(loss)
            # Make the optimizer step in a usual manner
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    from matplotlib import pyplot as plt
    plt.plot(range(len(losses)), losses)
    plt.show()

    # Finally, after training is finished, store the learnt GNN parameters in the file specified
    torch.save(gnn.parameters(), '../GNN_training/' + output_filename)


def main():
    generate_gnn_training_parameters('training_dataset.pkl', 'cifar_base_kw', 0.01, 100, "learnt_gnn_parameters.pkl")


if __name__ == '__main__':
    main()
