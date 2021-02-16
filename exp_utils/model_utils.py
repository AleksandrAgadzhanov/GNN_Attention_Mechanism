from torch import nn
from plnn.modules import Flatten
from plnn.model import simplify_network
import pandas as pd
import torch


def cifar_model_m2():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(16 * 8 * 8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_model():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32 * 8 * 8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def cifar_model_deep():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8 * 8 * 8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model


def add_single_prop(layers, gt, cls):
    """
    gt: ground truth label
    cls: class we want to verify against
    """
    additional_lin_layer = nn.Linear(10, 1, bias=True)
    lin_weights = additional_lin_layer.weight.data
    lin_weights.fill_(0)
    lin_bias = additional_lin_layer.bias.data
    lin_bias.fill_(0)
    lin_weights[0, cls] = -1
    lin_weights[0, gt] = 1

    # verif_layers2 = flatten_layers(verif_layers1,[1,14,14])
    final_layers = [layers[-1], additional_lin_layer]
    final_layer = simplify_network(final_layers)
    verif_layers = layers[:-1] + final_layer
    for layer in verif_layers:
        for p in layer.parameters():
            p.requires_grad = False

    return verif_layers


def load_verified_data(model_name, cifar_test=None):
    """
    This function returns the lists of CIFAR images, true labels and image indices as well as the model corresponding to
    the images which are correctly classified by the model
    """
    model = load_trained_model(model_name)

    if cifar_test is None:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.225, 0.225, 0.225])
        cifar_test = datasets.CIFAR10('../cifardata/', train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor(), normalize]))

    # Containers of correctly verified images, their true labels and their indices in the CIFAR dataset
    verified_images = []
    verified_true_labels = []
    verified_image_indices = []

    for image_index in range(len(cifar_test)):
        image, true_label = cifar_test[image_index]
        image = image.unsqueeze(0)
        predicted_label = torch.max(model(image)[0], 0)[1].item()

        # Only append the image, its true label and its index to the output if the model predicts the label correctly
        if predicted_label == true_label:
            verified_images.append(image)
            verified_true_labels.append(true_label)
            verified_image_indices.append(image_index)

    return verified_images, verified_true_labels, verified_image_indices, model


def match_with_properties(properties_filename, verified_images, verified_true_labels, verified_image_indices):
    """
    This function intersects the images corresponding to the rows in this dataframe and the input ones and then returns
    the resulting lists of images, their true and test labels and epsilon values. Depending on whether the property
    dataset provided is the one for training or testing, this function returns the intersection of the provided images
    and those in the dataset or the intersection with correctly verified properties only respectively.
    """
    # Construct the path and load the properties DataFrame
    properties_filepath = '../cifar_exp/' + properties_filename
    properties_dataframe = pd.read_pickle(properties_filepath)

    # If the properties dataset is for testing, leave only the correctly verified properties
    if properties_filename == 'base_easy.pkl' or properties_filename == 'base_med.pkl' or \
            properties_filename == 'base_hard.pkl':
        properties_dataframe = properties_dataframe[(properties_dataframe['BSAT_KWOld'] == 'False') |
                                                    (properties_dataframe['BSAT_KW'] == 'False') |
                                                    (properties_dataframe['BSAT_gnnkwT'] == 'False') |
                                                    (properties_dataframe['GSAT'] == 'False') |
                                                    (properties_dataframe['BSAT_gnnkwTO'] == 'False')]

    # Sort the properties DataFrame by the Idx column for the purpose of easier debugging
    properties_dataframe = properties_dataframe.sort_values(by=['Idx'], ascending=True)

    # Find the intersection of the set of images provided and of those appearing in the properties dataset and retrieve
    # all the required information from it
    images, true_labels, test_labels, epsilons = intersection(properties_dataframe, verified_images,
                                                              verified_true_labels, verified_image_indices)

    return images, true_labels, test_labels, epsilons


def intersection(properties_dataframe, verified_images, verified_true_labels, verified_image_indices):
    """
    This function takes a Dataframe of properties as well as lists of correctly verified images, their true labels and
    their images in the CIFAR dataset and then performs intersection of the sets of correctly verified images and those
    appearing in the Dataframe based on the indices. It then returns the lists of the images in the intersection set,
    their true and test labels and their respective epsilon values.
    """
    # Initialise the lists which will contain the outputs
    images = []
    true_labels = []
    test_labels = []
    epsilons = []

    # Now go over each row of the Dataframe
    for row_idx in range(len(properties_dataframe)):
        # If the index of the image appears in both the Dataframe and in the correctly verified image indices list, then
        # append the image and its true label to the respective output lists
        properties_dataframe_image_index = properties_dataframe.iloc[row_idx]['Idx']
        if properties_dataframe_image_index in verified_image_indices:
            verified_image_index = verified_image_indices.index(properties_dataframe_image_index)
            images.append(verified_images[verified_image_index])
            true_labels.append(verified_true_labels[verified_image_index])
            test_labels.append(properties_dataframe.iloc[row_idx]['prop'])
            epsilons.append(properties_dataframe.iloc[row_idx]['Eps'])

    return images, true_labels, test_labels, epsilons


def load_trained_model(model_name):
    """
    This function returns the required trained model based on the name provided as input.
    """
    if model_name == 'cifar_base_kw':
        model_path = 'models/cifar_base_kw.pth'
        model = cifar_model_m2()
        model.load_state_dict(torch.load(model_path, map_location="cpu")['state_dict'][0])
    elif model_name == 'cifar_wide_kw':
        model_path = '../models/cifar_wide_kw.pth'
        model = cifar_model()
        model.load_state_dict(torch.load(model_path, map_location="cpu")['state_dict'][0])
    elif model_name == 'cifar_deep_kw':
        model_path = '../models/cifar_deep_kw.pth'
        model = cifar_model_deep()
        model.load_state_dict(torch.load(model_path, map_location="cpu")['state_dict'][0])
    else:
        raise NotImplementedError

    return model
