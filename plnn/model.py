from torch import nn
from plnn.modules import View
import torch


def simplify_network(all_layers):
    """
    Given a sequence of Pytorch nn.Module `all_layers`,
    representing a feed-forward neural network,
    merge the layers when two sucessive modules are nn.Linear
    and can therefore be equivalenty computed as a single nn.Linear
    """
    new_all_layers = [all_layers[0]]
    for layer in all_layers[1:]:
        if (type(layer) is nn.Linear) and (type(new_all_layers[-1]) is nn.Linear):
            # We can fold together those two layers
            prev_layer = new_all_layers.pop()

            joint_weight = torch.mm(layer.weight.data, prev_layer.weight.data)
            if prev_layer.bias is not None:
                joint_bias = layer.bias.data + torch.mv(layer.weight.data, prev_layer.bias.data)
            else:
                joint_bias = layer.bias.data

            joint_out_features = layer.out_features
            joint_in_features = prev_layer.in_features

            joint_layer = nn.Linear(joint_in_features, joint_out_features)
            joint_layer.bias.data.copy_(joint_bias)
            joint_layer.weight.data.copy_(joint_weight)
            new_all_layers.append(joint_layer)
        elif (type(layer) is nn.MaxPool1d) and (layer.kernel_size == 1) and (layer.stride == 1):
            # This is just a spurious Maxpooling because the kernel_size is 1
            # We will do nothing
            pass
        elif (type(layer) is View) and (type(new_all_layers[-1]) is View):
            # No point in viewing twice in a row
            del new_all_layers[-1]

            # Figure out what was the last thing that imposed a shape
            # and if this shape was the proper one.
            prev_layer_idx = -1
            lay_nb_dim_inp = 0
            while True:
                parent_lay = new_all_layers[prev_layer_idx]
                prev_layer_idx -= 1
                if type(parent_lay) is nn.ReLU:
                    # Can't say anything, ReLU is flexible in dimension
                    continue
                elif type(parent_lay) is nn.Linear:
                    lay_nb_dim_inp = 1
                    break
                elif type(parent_lay) is nn.MaxPool1d:
                    lay_nb_dim_inp = 2
                    break
                else:
                    raise NotImplementedError
            if len(layer.out_shape) != lay_nb_dim_inp:
                # If the View is actually necessary, add the change
                new_all_layers.append(layer)
                # Otherwise do nothing
        else:
            new_all_layers.append(layer)
    return new_all_layers
