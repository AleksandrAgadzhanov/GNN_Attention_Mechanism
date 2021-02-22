import torch
import copy
from plnn.proxlp_solver.solver import SaddleLP


def _run_lp_as_init(verif_layers, domain, device='cpu'):
    decision_bound = 0

    if device == 'cuda' and torch.cuda.is_available():
        cuda_verif_layers = [copy.deepcopy(lay).cuda() for lay in verif_layers]
        domain = domain.cuda()
    else:
        cuda_verif_layers = [copy.deepcopy(lay) for lay in verif_layers]

    # use best of naive interval propagation and KW as intermediate bounds
    intermediate_net = SaddleLP(cuda_verif_layers, store_bounds_primal=True)
    intermediate_net.set_solution_optimizer('best_naive_kw', None)

    bounds_net = SaddleLP(cuda_verif_layers, store_bounds_primal=True, store_duals=True)
    bounds_net.set_decomposition('pairs', 'KW')
    adam_params = {
        'nb_steps': 100,
        'initial_step_size': 1e-2,
        'final_step_size': 1e-3,
        'betas': (0.9, 0.999),
        'outer_cutoff': 0,
        'log_values': False
    }
    bounds_net.set_solution_optimizer('adam', adam_params)
    # bounds_net.set_solution_optimizer('best_naive_kw', None)

    # do initial computation for the network as it is (batch of size 1: there is only one domain)
    # get intermediate bounds
    intermediate_net.define_linear_approximation(domain.unsqueeze(0))
    intermediate_lbs = copy.deepcopy(intermediate_net.lower_bounds)
    intermediate_ubs = copy.deepcopy(intermediate_net.upper_bounds)

    # if intermediate_lbs[-1] > decision_bound or intermediate_ubs[-1] < decision_bound:
    #     # bab.join_children(gurobi_dict, timeout)
    #     return intermediate_lbs[-1], intermediate_ubs[-1], \
    #            intermediate_net.get_lower_bound_network_input(), nb_visited_states, fail_safe_ratio

    # print('computing last layer bounds')
    # compute last layer bounds with a more expensive network

    bounds_net.build_model_using_bounds(domain.unsqueeze(0), (intermediate_lbs, intermediate_ubs))

    global_lb, global_ub = bounds_net.compute_lower_bound(counterexample_verification=True)
    intermediate_lbs[-1] = global_lb
    intermediate_ubs[-1] = global_ub
    bounds_net_device = global_lb.device
    intermediate_net_device = domain.device

    # retrieve bounds info from the bounds network
    global_ub_point = bounds_net.get_lower_bound_network_input()
    global_ub = bounds_net.net(global_ub_point)

    duals_ = bounds_net.get_dual_vars().rhos

    return global_ub_point, global_ub, intermediate_lbs, intermediate_ubs, duals_
