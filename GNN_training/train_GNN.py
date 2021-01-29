import torch
from exp_utils.model_utils import load_trained_model
from GNN_framework.GraphNeuralNetwork import GraphNeuralNetwork
from GNN_framework.helper_functions import simplify_model, perturb_image, gradient_ascent


def generate_gnn_training_parameters(training_dataset_filename, model_name, gnn_learning_rate, pgd_learning_rate,
                                     num_iterations, num_epochs, output_filename):
    """
    This function performs the actual training of the specified Graph Neural Network. It accepts the GraphNeuralNetwork
    object, the filename of the training dataset as well as some other parameters as inputs and stores the learned model
    parameters in a separate file.
    """
    # First, load the training dataset which is a list of feature dictionaries from the specified filename. Also load
    # the model
    list_of_feature_dicts = torch.load('../GNN_training/' + training_dataset_filename)
    model = load_trained_model(model_name)

    # Create the temporary variables which will only be used to initialise the GNN structure. Then create an instance of
    # the GraphNeuralNetwork object using these variables
    temp_simplified_model = simplify_model(model, list_of_feature_dicts[0]['true label'],
                                           list_of_feature_dicts[0]['test label'])
    temp_input_size = list_of_feature_dicts[0]['input size']
    temp_input_feature_size = list_of_feature_dicts[0]['input'].size()[0]
    temp_relu_feature_size = list_of_feature_dicts[0]['hidden'][0].size()[0]
    temp_output_feature_size = list_of_feature_dicts[0]['output'].size()[0]
    gnn = GraphNeuralNetwork(temp_simplified_model, temp_input_size, temp_input_feature_size, temp_relu_feature_size,
                             temp_output_feature_size, training_mode=True)

    # Initialise the optimizer on the parameters of the GNN
    optimizer = torch.optim.Adam(gnn.parameters(), lr=gnn_learning_rate)

    # Follow the training algorithm for a specified number of epochs
    for epoch in range(num_epochs):
        # For each subdomain appearing in the training dataset
        for subdomain_index in range(len(list_of_feature_dicts)):
            feature_dict = list_of_feature_dicts[subdomain_index]

            # Simplify the model according to the current true and test class labels
            simplified_model = simplify_model(model, feature_dict['true label'], feature_dict['test label'])

            # When the epoch or the subdomain index are not the first one, reset the input embedding vectors since the
            # forward input update function only activates when the input embedding vectors are zero
            if epoch != 0 or subdomain_index != 0:
                gnn.reset_input_embedding_vectors()

            # Perform a series of forward and backward updates of all the embedding vectors within the GNN
            gnn.update_embedding_vectors(feature_dict['input'], feature_dict['hidden'], feature_dict['output'])

            # Update the domain bounds for each pixel based on the GNN outputs
            old_lower_bound = feature_dict['input'][0, :].reshape(temp_input_size)
            old_upper_bound = feature_dict['input'][1, :].reshape(temp_input_size)
            new_lower_bound, new_upper_bound = gnn.compute_updated_bounds(old_lower_bound, old_upper_bound)

            # Perturb each pixel within the updated domain bounds
            perturbed_image = perturb_image(new_lower_bound, new_upper_bound)

            # Perform a PGD attack given the new bounds and perturbation
            loss = gradient_ascent(simplified_model, perturbed_image, new_lower_bound, new_upper_bound,
                                   pgd_learning_rate, num_iterations, return_loss=True)

            # Make the optimizer step in a usual manner
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main():
    generate_gnn_training_parameters('training_dataset.pkl', 'cifar_base_kw', 0.1, 0.1, 10, 2, '')


if __name__ == '__main__':
    main()
