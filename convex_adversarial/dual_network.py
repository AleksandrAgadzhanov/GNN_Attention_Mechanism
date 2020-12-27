import torch
import torch.nn as nn
from convex_adversarial.dual_layers import DualReLU

from .utils import Dense, DenseSequential
from .dual_inputs import select_input
from .dual_layers import select_layer


class DualNetwork(nn.Module):
    def __init__(self, net, X, epsilon,
                 proj=None, norm_type='l1', bounded_input=False,
                 data_parallel=True, mask=None, provided_zl=None, provided_zu=None):
        """
        This class creates the dual network.
        net : ReLU network
        X : minibatch of examples
        epsilon : size of l1 norm ball to be robust against adversarial examples
        alpha_grad : flag to propagate gradient through alpha
        scatter_grad : flag to propagate gradient through scatter operation
        l1 : size of l1 projection
        l1_eps : the bound is correct up to a 1/(1-l1_eps) factor
        m : number of probabilistic bounds to take the max over
        """
        super(DualNetwork, self).__init__()
        # need to change that if no batchnorm, can pass just a single example
        if not isinstance(net, (nn.Sequential, DenseSequential)):
            raise ValueError("Network must be a nn.Sequential or DenseSequential module")
        with torch.no_grad():
            if any('BatchNorm2d' in str(l.__class__.__name__) for l in net):
                zs = [X]
            else:
                zs = [X[:1]]
            nf = [zs[0].size()]
            for l in net:
                if isinstance(l, Dense):
                    zs.append(l(*zs))
                else:
                    zs.append(l(zs[-1]))
                nf.append(zs[-1].size())

        # self.nf = nf

        # Use the bounded boxes
        dual_net = [select_input(X, epsilon, proj, norm_type, bounded_input)]

        # change_bounds=False
        # if mask is not None:
        #    change_bounds = True
        replace_bounds = False
        if provided_zl is not None and provided_zu is not None:
            replace_bounds = True
            provided_layers_length = len(provided_zu)

        # elif provided_zl is None and provided_zu is None:
        #    pass
        # else:
        #    print('must provide both variables: zu, zl')

        mask_idx = 0
        for i, (in_f, out_f, layer) in enumerate(zip(nf[:-1], nf[1:], net)):
            dual_layer = select_layer(layer, dual_net, X, proj, norm_type,
                                      in_f, out_f, zs[i])
            if isinstance(dual_layer, DualReLU) and replace_bounds:
                #    if change_bounds:
                #        temp = mask[mask_idx]
                #        temp = temp.reshape(dual_layer.zu.size())
                #        # change the ub that is positive  to 0 if the corresponding
                #        # mask decision is 0
                #        #import pdb; pdb.set_trace()
                #        dual_layer.zu = -F.relu(-dual_layer.zu*(temp==0).float()) + F.relu(dual_layer.zu*(temp!=0).float())
                #        # change the lb that is negative to 0 if the corresponding
                #        # mask decision is 1
                #        dual_layer.zl = F.relu(dual_layer.zl*(temp==1).float()) - F.relu(-dual_layer.zl*(temp!=1).float())
                #
                zu_pre = dual_layer.zu
                zl_pre = dual_layer.zl

                #    if replace_bounds:
                #        #import pdb; pdb.set_trace()
                zu_pre = torch.min(zu_pre, provided_zu[mask_idx])
                zl_pre = torch.max(zl_pre, provided_zl[mask_idx])

                dual_layer = select_layer(layer, dual_net, X, proj, norm_type, in_f, out_f, zs[i], zl=zl_pre, zu=zu_pre)
                mask_idx += 1

            # skip last layer
            if i < len(net) - 1:
                for l in dual_net:
                    l.apply(dual_layer)
                dual_net.append(dual_layer)
            else:
                self.last_layer = dual_layer

        self.dual_net = dual_net
        return

    def forward(self, c):
        """ For the constructed given dual network, compute the objective for
        some given vector c """
        nu = [-c]
        nu.append(self.last_layer.T(*nu))
        for l in reversed(self.dual_net[1:]):
            nu.append(l.T(*nu))
        dual_net = self.dual_net + [self.last_layer]

        # interm = [l.objective(*nu[:min(len(dual_net)-i+1, len(dual_net))]) for
        #        i,l in enumerate(dual_net)]
        # print(interm)
        # import pdb
        # pdb.set_trace()

        return sum(l.objective(*nu[:min(len(dual_net) - i + 1, len(dual_net))]) for
                   i, l in enumerate(dual_net))