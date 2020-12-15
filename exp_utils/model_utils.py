from torch import nn
from plnn.modules import Flatten
from plnn.model import simplify_network
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
    gt: ground truth lable
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


def load_cifar_pgd_exp(model, cifar_test=None):
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

    # Containers of correctly verified images, their true labels and their indices in the CIFAR dataset
    images = []
    true_labels = []
    image_indices = []

    for image_index in range(len(cifar_test)):
        x, y = cifar_test[image_index]
        x = x.unsqueeze(0)
        y_pred = torch.max(model(x)[0], 0)[1].item()

        # Only append the image, its true label and its index to the output if the model predicts the label correctly
        if y_pred == y:
            images.append(x)
            true_labels.append(y)
            image_indices.append(image_index)

    return images, true_labels, image_indices, model
