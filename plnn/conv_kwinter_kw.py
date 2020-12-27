import gurobipy as grb
import torch
from plnn.dual_network_linear_approximation import LooseDualNetworkApproximation
from plnn.modules import View, Flatten
from torch.nn import functional as F
from torch import nn


def build_the_model(layers, input_domain, x, ball_eps, bounded):
    """
    Before the first branching, we build the model and create a mask matrix

    Output: relu_mask, current intermediate upper and lower bounds, a list of indices of the layers right before a Relu
            layer the constructed gurobi model

    NOTE: we keep all bounds as a list of tensors from now on. Only lower and upper bounds are kept in the same shape as
          layers' outputs. Mask is linearized Gurobi_var lists are linearized model_lower_bounds and model_upper_bounds
          are kept mainly for debugging purpose and could be removed
    """
    new_relu_mask = []
    lower_bounds = []
    upper_bounds = []

    # NEW STRUCTURE: deal with all available bounds first get KW bounds
    loose_dual = LooseDualNetworkApproximation(layers, x, ball_eps)
    kw_lb, kw_ub, pre_relu_indices, dual_info = loose_dual.init_kw_bounds(bounded)

    # second get interval bounds
    if len(input_domain.size()) == 2:
        lower_bounds.append(input_domain[:, 0].squeeze(-1))
        upper_bounds.append(input_domain[:, 1].squeeze(-1))
    else:
        lower_bounds.append(input_domain[:, :, :, 0].squeeze(-1))
        upper_bounds.append(input_domain[:, :, :, 1].squeeze(-1))
    layer_idx = 1
    for layer in layers:
        if type(layer) is nn.Linear:
            pre_lb = lower_bounds[-1]
            pre_ub = upper_bounds[-1]
            pos_w = torch.clamp(layer.weight, 0, None)
            neg_w = torch.clamp(layer.weight, None, 0)
            out_lbs = pos_w @ pre_lb + neg_w @ pre_ub + layer.bias
            out_ubs = pos_w @ pre_ub + neg_w @ pre_lb + layer.bias
            # Get the better estimates from KW and Interval Bounds
            new_layer_lb = torch.max(kw_lb[layer_idx], out_lbs)
            new_layer_ub = torch.min(kw_ub[layer_idx], out_ubs)
        elif type(layer) is nn.Conv2d:
            assert layer.dilation == (1, 1)
            pre_lb = lower_bounds[-1].unsqueeze(0)
            pre_ub = upper_bounds[-1].unsqueeze(0)
            pos_weight = torch.clamp(layer.weight, 0, None)
            neg_weight = torch.clamp(layer.weight, None, 0)

            out_lbs = (F.conv2d(pre_lb, pos_weight, layer.bias,
                                layer.stride, layer.padding, layer.dilation, layer.groups)
                       + F.conv2d(pre_ub, neg_weight, None,
                                  layer.stride, layer.padding, layer.dilation, layer.groups))
            out_ubs = (F.conv2d(pre_ub, pos_weight, layer.bias,
                                layer.stride, layer.padding, layer.dilation, layer.groups)
                       + F.conv2d(pre_lb, neg_weight, None,
                                  layer.stride, layer.padding, layer.dilation, layer.groups))

            new_layer_lb = (torch.max(kw_lb[layer_idx], out_lbs)).squeeze(0)
            new_layer_ub = (torch.min(kw_ub[layer_idx], out_ubs)).squeeze(0)
        elif type(layer) == nn.ReLU:
            new_layer_lb = F.relu(lower_bounds[-1])
            new_layer_ub = F.relu(upper_bounds[-1])

        elif type(layer) == View:
            continue
        elif type(layer) == Flatten:
            new_layer_lb = lower_bounds[-1].view(-1)
            new_layer_ub = upper_bounds[-1].view(-1)
        else:
            raise NotImplementedError

        lower_bounds.append(new_layer_lb)
        upper_bounds.append(new_layer_ub)

        layer_idx += 1

    # compare KW_INT bounds with KW bounds. if they are different, re-update the kw model
    for pre_idx in pre_relu_indices:
        if torch.sum(abs(lower_bounds[pre_idx] - kw_lb[pre_idx]) > 1e-4) == 0 and torch.sum(
                abs(upper_bounds[pre_idx] - kw_ub[pre_idx]) > 1e-4) == 0:
            pass
        else:
            print(f"initial kw: change_idx at {pre_idx}")
            lower_bounds, upper_bounds, dual_info = loose_dual.update_kw_bounds(pre_idx, pre_lb_all=lower_bounds,
                                                                                pre_ub_all=upper_bounds,
                                                                                dual_info=dual_info)
            break

    # record the dual_info as an attribute of the loose_dual instance this should be the only dual instance recorded and
    # should not be modified
    loose_dual.orig_dual = dual_info

    # NEW STRUCTURE: use the computed bounds to directly introduce gurobi models

    # Initialize the model
    model = grb.Model()
    model.setParam('OutputFlag', False)
    model.setParam('Threads', 1)
    # keep a record of model's information
    gurobi_vars = []

    # Do the input layer, which is a special case
    inp_gurobi_vars = []
    zero_var = model.addVar(lb=0, ub=0, obj=0, vtype=grb.GRB.CONTINUOUS, name='zero')
    if input_domain.dim() == 2:
        # This is a linear input.
        for dim, (lb, ub) in enumerate(input_domain):
            v = model.addVar(lb=lb, ub=ub, obj=0, vtype=grb.GRB.CONTINUOUS, name=f'inp_{dim}')
            inp_gurobi_vars.append(v)
    else:
        assert input_domain.dim() == 4
        for chan in range(input_domain.size(0)):
            chan_vars = []
            for row in range(input_domain.size(1)):
                row_vars = []
                for col in range(input_domain.size(2)):
                    lb = input_domain[chan, row, col, 0]
                    ub = input_domain[chan, row, col, 1]
                    v = model.addVar(lb=lb, ub=ub, obj=0, vtype=grb.GRB.CONTINUOUS, name=f'inp_[{chan},{row},{col}]')
                    row_vars.append(v)
                chan_vars.append(row_vars)
            inp_gurobi_vars.append(chan_vars)
    model.update()

    gurobi_vars.append(inp_gurobi_vars)

    # Do the other layers, computing for each of the neuron, its upper bound and lower bound
    layer_idx = 1
    for layer in layers:
        new_layer_gurobi_vars = []
        if type(layer) is nn.Linear:
            # Get the better estimates from KW and Interval Bounds
            out_lbs = lower_bounds[layer_idx]
            out_ubs = upper_bounds[layer_idx]
            for neuron_idx in range(layer.weight.size(0)):
                lin_expr = layer.bias[neuron_idx].item()
                coeffs = layer.weight[neuron_idx, :]
                lin_expr += grb.LinExpr(coeffs, gurobi_vars[-1])

                out_lb = out_lbs[neuron_idx].item()
                out_ub = out_ubs[neuron_idx].item()
                v = model.addVar(lb=out_lb, ub=out_ub, obj=0, vtype=grb.GRB.CONTINUOUS,
                                 name=f'lay{layer_idx}_{neuron_idx}')
                model.addConstr(v == lin_expr)
                model.update()

                new_layer_gurobi_vars.append(v)

        elif type(layer) is nn.Conv2d:
            assert layer.dilation == (1, 1)
            pre_lb_size = lower_bounds[layer_idx - 1].unsqueeze(0).size()
            out_lbs = lower_bounds[layer_idx].unsqueeze(0)
            out_ubs = upper_bounds[layer_idx].unsqueeze(0)

            for out_chan_idx in range(out_lbs.size(1)):
                out_chan_vars = []
                for out_row_idx in range(out_lbs.size(2)):
                    out_row_vars = []
                    for out_col_idx in range(out_lbs.size(3)):
                        lin_expr = layer.bias[out_chan_idx].item()

                        for in_chan_idx in range(layer.weight.shape[1]):
                            for ker_row_idx in range(layer.weight.shape[2]):
                                in_row_idx = -layer.padding[0] + layer.stride[0] * out_row_idx + ker_row_idx
                                if (in_row_idx < 0) or (in_row_idx == pre_lb_size[2]):
                                    # This is padding -> value of 0
                                    continue
                                for ker_col_idx in range(layer.weight.shape[3]):
                                    in_col_idx = -layer.padding[1] + layer.stride[1] * out_col_idx + ker_col_idx
                                    if (in_col_idx < 0) or (in_col_idx == pre_lb_size[3]):
                                        # This is padding -> value of 0
                                        continue
                                    coeff = layer.weight[out_chan_idx, in_chan_idx, ker_row_idx, ker_col_idx].item()
                                    lin_expr += coeff * gurobi_vars[-1][in_chan_idx][in_row_idx][in_col_idx]

                        out_lb = out_lbs[0, out_chan_idx, out_row_idx, out_col_idx].item()
                        out_ub = out_ubs[0, out_chan_idx, out_row_idx, out_col_idx].item()
                        v = model.addVar(lb=out_lb, ub=out_ub, obj=0, vtype=grb.GRB.CONTINUOUS,
                                         name=f'lay{layer_idx}_[{out_chan_idx}, {out_row_idx}, {out_col_idx}]')
                        model.addConstr(v == lin_expr)
                        model.update()

                        out_row_vars.append(v)
                    out_chan_vars.append(out_row_vars)
                new_layer_gurobi_vars.append(out_chan_vars)

        elif type(layer) == nn.ReLU:
            if isinstance(gurobi_vars[-1][0], list):
                # This is convolutional
                pre_ubs = upper_bounds[layer_idx - 1]
                new_layer_mask = []
                ratios_all = dual_info[0][layer_idx].d
                for chan_idx, channel in enumerate(gurobi_vars[-1]):
                    chan_vars = []
                    for row_idx, row in enumerate(channel):
                        row_vars = []
                        for col_idx, pre_var in enumerate(row):
                            slope = ratios_all[0, chan_idx, row_idx, col_idx].item()
                            pre_ub = pre_ubs[chan_idx, row_idx, col_idx].item()

                            if slope == 1.0:
                                # ReLU is always passing
                                v = pre_var
                                new_layer_mask.append(1)
                            elif slope == 0.0:
                                v = zero_var
                                new_layer_mask.append(0)
                            else:
                                lb = 0
                                ub = pre_ub
                                new_layer_mask.append(-1)
                                v = model.addVar(lb=lb, ub=ub, obj=0, vtype=grb.GRB.CONTINUOUS,
                                                 name=f'ReLU{layer_idx}_[{chan_idx},{row_idx},{col_idx}]')
                            row_vars.append(v)
                        chan_vars.append(row_vars)
                    new_layer_gurobi_vars.append(chan_vars)
            else:
                pre_ubs = upper_bounds[layer_idx - 1]
                new_layer_mask = []
                ratios_all = dual_info[0][layer_idx].d.squeeze(0)
                assert isinstance(gurobi_vars[-1][0], grb.Var)
                for neuron_idx, pre_var in enumerate(gurobi_vars[-1]):
                    pre_ub = pre_ubs[neuron_idx].item()
                    slope = ratios_all[neuron_idx].item()

                    if slope == 1.0:
                        # The ReLU is always passing
                        v = pre_var
                        new_layer_mask.append(1)
                    elif slope == 0.0:
                        v = zero_var
                        # No need to add an additional constraint that v==0 because this will be covered by the bounds
                        # we set on the value of v
                        new_layer_mask.append(0)
                    else:
                        lb = 0
                        ub = pre_ub
                        v = model.addVar(lb=lb, ub=ub, obj=0, vtype=grb.GRB.CONTINUOUS,
                                         name=f'ReLU{layer_idx}_{neuron_idx}')
                        new_layer_mask.append(-1)

                    new_layer_gurobi_vars.append(v)

            new_relu_mask.append(torch.tensor(new_layer_mask))

        elif type(layer) == View:
            continue
        elif type(layer) == Flatten:
            for chan_idx in range(len(gurobi_vars[-1])):
                for row_idx in range(len(gurobi_vars[-1][chan_idx])):
                    new_layer_gurobi_vars.extend(gurobi_vars[-1][chan_idx][row_idx])
        else:
            raise NotImplementedError

        gurobi_vars.append(new_layer_gurobi_vars)

        layer_idx += 1

    # Assert that this is as expected a network with a single output
    assert len(gurobi_vars[-1]) == 1, "Network doesn't have scalar output"

    model.update()

    model.setObjective(gurobi_vars[-1][0], grb.GRB.MINIMIZE)
    model.optimize()
    check_optimization_success(model)

    glb = gurobi_vars[-1][0].X
    lower_bounds[-1] = torch.tensor([glb])

    return new_relu_mask, lower_bounds, upper_bounds


class InfeasibleMaskException(Exception):
    pass


def check_optimization_success(model, introduced_constrs=None):
    if model.status == 2:
        # Optimization successful, nothing to complain about
        pass
    elif model.status == 3:
        model.remove(introduced_constrs)
        # The model is infeasible. We have made incompatible
        # assumptions, so this subdomain doesn't exist.
        raise InfeasibleMaskException()
    else:
        print('\n')
        print(f'model.status: {model.status}\n')
        raise NotImplementedError
