from torch import nn
from plnn.modules import View
from plnn.naive_approximation import NaiveNetwork
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


def reluify_maxpool(layers, domain):
    '''
    Remove all the Maxpool units of a feedforward network represented by
    `layers` and replace them by an equivalent combination of ReLU + Linear
    This is only valid over the domain `domain` because we use some knowledge
    about upper and lower bounds of certain neurons
    '''
    naive_net = NaiveNetwork(layers)
    naive_net.do_interval_analysis(domain)
    lbs = naive_net.lower_bounds
    layers = layers[:]

    new_all_layers = []

    idx_of_inp_lbs = 0
    layer_idx = 0
    while layer_idx < len(layers):
        layer = layers[layer_idx]
        if type(layer) is nn.MaxPool1d:
            # We need to decompose this MaxPool until it only has a size of 2
            assert layer.padding == 0
            assert layer.dilation == 1
            if layer.kernel_size > 2:
                assert layer.kernel_size % 2 == 0, "Not supported yet"
                assert layer.stride % 2 == 0, "Not supported yet"
                # We're going to decompose this maxpooling into two maxpooling
                # max(     in_1, in_2 ,      in_3, in_4)
                # will become
                # max( max(in_1, in_2),  max(in_3, in_4))
                first_mp = nn.MaxPool1d(2, stride=2)
                second_mp = nn.MaxPool1d(layer.kernel_size // 2,
                                         stride=layer.stride // 2)
                # We will replace the Maxpooling that was originally there with
                # those two layers
                # We need to add a corresponding layer of lower bounds
                first_lbs = lbs[idx_of_inp_lbs]
                intermediate_lbs = []
                for pair_idx in range(len(first_lbs) // 2):
                    intermediate_lbs.append(max(first_lbs[2*pair_idx],
                                                first_lbs[2*pair_idx+1]))
                # Do the replacement
                del layers[layer_idx]
                layers.insert(layer_idx, first_mp)
                layers.insert(layer_idx+1, second_mp)
                lbs.insert(idx_of_inp_lbs+1, intermediate_lbs)

                # Now continue so that we re-go through the loop with the now
                # simplified maxpool
                continue
            elif layer.kernel_size == 2:
                # Each pair need two in the intermediate layers that is going
                # to be Relu-ified
                pre_nb_inp_lin = len(lbs[idx_of_inp_lbs])
                # How many starting position can we fit in?
                # 1 + how many stride we can fit before we're too late in the array to fit a kernel_size
                pre_nb_out_lin = (1 + ((pre_nb_inp_lin - layer.kernel_size) // layer.stride)) * 2
                pre_relu_lin = nn.Linear(pre_nb_inp_lin, pre_nb_out_lin, bias=True)
                pre_relu_weight = pre_relu_lin.weight.data
                pre_relu_bias = pre_relu_lin.bias.data
                pre_relu_weight.zero_()
                pre_relu_bias.zero_()
                # For each of (x, y) that needs to be transformed to max(x, y)
                # We create (x-y, y-y_lb)
                first_in_index = 0
                first_out_index = 0
                while first_in_index + 1 < pre_nb_inp_lin:
                    pre_relu_weight[first_out_index, first_in_index] = 1
                    pre_relu_weight[first_out_index, first_in_index+1] = -1

                    pre_relu_weight[first_out_index+1, first_in_index+1] = 1
                    pre_relu_bias[first_out_index+1] = -lbs[idx_of_inp_lbs][first_in_index + 1]

                    # Now shift
                    first_in_index += layer.stride
                    first_out_index += 2
                new_all_layers.append(pre_relu_lin)
                new_all_layers.append(nn.ReLU())

                # We now need to create the second layer
                # It will sum [max(x-y, 0)], [max(y - y_lb, 0)] and y_lb
                post_nb_inp_lin = pre_nb_out_lin
                post_nb_out_lin = post_nb_inp_lin // 2
                post_relu_lin = nn.Linear(post_nb_inp_lin, post_nb_out_lin)
                post_relu_weight = post_relu_lin.weight.data
                post_relu_bias = post_relu_lin.bias.data
                post_relu_weight.zero_()
                post_relu_bias.zero_()
                first_in_index = 0
                out_index = 0
                while first_in_index + 1 < post_nb_inp_lin:
                    post_relu_weight[out_index, first_in_index] = 1
                    post_relu_weight[out_index, first_in_index+1] = 1
                    post_relu_bias[out_index] = lbs[idx_of_inp_lbs][layer.stride*out_index+1]
                    first_in_index += 2
                    out_index += 1
                new_all_layers.append(post_relu_lin)
                idx_of_inp_lbs += 1
            else:
                # This should have been cleaned up in one of the simplify passes
                raise NotImplementedError
        elif type(layer) is nn.Linear:
            new_all_layers.append(layer)
            idx_of_inp_lbs += 1
        elif type(layer) is nn.ReLU:
            new_all_layers.append(layer)
        elif type(layer) is View:
            # We shouldn't add the view as we are getting rid of them
            pass
        layer_idx += 1
    return new_all_layers
