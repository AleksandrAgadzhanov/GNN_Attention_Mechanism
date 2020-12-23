from torch import nn
from plnn.modules import Flatten
from plnn.model import simplify_network
import pandas as pd
import torch
import random


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


def load_cifar_1to1_exp(model, idx, test=None, cifar_test=None):
    if model == 'cifar_base_kw':
        model_name = './models/cifar_base_kw.pth'
        model = cifar_model_m2()
        model.load_state_dict(torch.load(model_name, map_location="cpu")['state_dict'][0])
    elif model == 'cifar_wide_kw':
        model_name = './models/cifar_wide_kw.pth'
        model = cifar_model()
        model.load_state_dict(torch.load(model_name, map_location="cpu")['state_dict'][0])
    elif model == 'cifar_deep_kw':
        model_name = './models/cifar_deep_kw.pth'
        model = cifar_model_deep()
        model.load_state_dict(torch.load(model_name, map_location="cpu")['state_dict'][0])
    else:
        raise NotImplementedError

    if cifar_test is None:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.225, 0.225, 0.225])
        cifar_test = datasets.CIFAR10('./cifardata/', train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor(), normalize]))

    x, y = cifar_test[idx]
    x = x.unsqueeze(0)
    # first check the model is correct at the input
    y_pred = torch.max(model(x)[0], 0)[1].item()
    print('predicted label ', y_pred, ' correct label ', y)
    if y_pred != y:
        print('model prediction is incorrect for the given model')
        return None, None, None
    else:
        if test is None:
            choices = list(range(10))
            choices.remove(y_pred)
            test = random.choice(choices)

        print('tested against ', test)
        for p in model.parameters():
            p.requires_grad = False

        layers = list(model.children())
        added_prop_layers = add_single_prop(layers, y_pred, test)
        return x, added_prop_layers, test


def load_cifar_data(model, cifar_test=None):
    """
    This function returns the lists of CIFAR images, true labels and image indices as well as the model corresponding to
    the images which are correctly classified by the model
    """
    if model == 'cifar_base_kw':
        model_name = '../models/cifar_base_kw.pth'
        model = cifar_model_m2()
        model.load_state_dict(torch.load(model_name, map_location="cpu")['state_dict'][0])
    elif model == 'cifar_wide_kw':
        model_name = '../models/cifar_wide_kw.pth'
        model = cifar_model()
        model.load_state_dict(torch.load(model_name, map_location="cpu")['state_dict'][0])
    elif model == 'cifar_deep_kw':
        model_name = '../models/cifar_deep_kw.pth'
        model = cifar_model_deep()
        model.load_state_dict(torch.load(model_name, map_location="cpu")['state_dict'][0])
    else:
        raise NotImplementedError

    if cifar_test is None:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.225, 0.225, 0.225])
        cifar_test = datasets.CIFAR10('../cifardata/', train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor(), normalize]))

    # Containers of correctly verified images, their true labels and their indices in the CIFAR dataset
    images = []
    true_labels = []
    image_indices = []

    for image_index in range(len(cifar_test)):
        image, true_label = cifar_test[image_index]
        image = image.unsqueeze(0)
        predicted_label = torch.max(model(image)[0], 0)[1].item()

        # Only append the image, its true label and its index to the output if the model predicts the label correctly
        if predicted_label == true_label:
            images.append(image)
            true_labels.append(true_label)
            image_indices.append(image_index)

    return images, true_labels, image_indices, model


def load_properties_data(properties_filename, images, true_labels, image_indices):
    """
    This function processes the properties dataframe first, leaving only the properties which were correctly verified,
    and then returns the lists of images, true labels, test labels and epsilons which correspond to these properties
    only
    """
    # Load the properties DataFrame, leave only verified properties
    properties_filepath = '../cifar_exp/' + properties_filename
    properties_dataframe = pd.read_pickle(properties_filepath)
    properties_dataframe = properties_dataframe[(properties_dataframe['BSAT_KWOld'] == 'False') |
                                                (properties_dataframe['BSAT_KW'] == 'False') |
                                                (properties_dataframe['BSAT_gnnkwT'] == 'False') |
                                                (properties_dataframe['GSAT'] == 'False') |
                                                (properties_dataframe['BSAT_gnnkwTO'] == 'False')]

    # Remove the single property which has the same index as one of the other properties but a different prop value
    # (applies to base_easy.pkl only)
    if properties_filename == 'base_easy.pkl':
        properties_dataframe = properties_dataframe.drop(properties_dataframe[(properties_dataframe['Idx'] == 6100) &
                                                                              (properties_dataframe[
                                                                                   'prop'] == 9)].index)

    # Sort the properties DataFrame by the Idx column for the purpose of easier debugging
    properties_dataframe = properties_dataframe.sort_values(by=['Idx'], ascending=True)

    # Drop all the elements of the images, true_labels and image_indices which do not appear in the properties file
    properties_image_indices = list(properties_dataframe['Idx'])
    array_length = len(image_indices)
    for i in range(array_length - 1, -1, -1):  # counter starts at the end due to the nature of the pop() function
        if image_indices[i] not in properties_image_indices:
            images.pop(i)
            true_labels.pop(i)
            image_indices.pop(i)

    # Create the list of classes the properties were verified against and the list of epsilons
    test_labels = []
    epsilons = []
    for i in range(len(images)):
        test_labels.append(properties_dataframe[properties_dataframe['Idx'] == image_indices[i]].iloc[0]['prop'])
        epsilons.append(properties_dataframe[properties_dataframe['Idx'] == image_indices[i]].iloc[0]['Eps'])

    return images, true_labels, test_labels, epsilons
