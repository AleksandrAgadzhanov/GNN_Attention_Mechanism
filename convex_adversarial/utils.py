import torch.nn as nn


class DenseSequential(nn.Sequential):
    def forward(self, x):
        xs = [x]
        for module in self._modules.values():
            if 'Dense' in type(module).__name__:
                xs.append(module(*xs))
            else:
                xs.append(module(xs[-1]))
        return xs[-1]

class Dense(nn.Module):
    def __init__(self, *Ws):
        super(Dense, self).__init__()
        self.Ws = nn.ModuleList(list(Ws))
        if len(Ws) > 0 and hasattr(Ws[0], 'out_features'):
            self.out_features = Ws[0].out_features

    def forward(self, *xs):
        xs = xs[-len(self.Ws):]
        out = sum(W(x) for x,W in zip(xs, self.Ws) if W is not None)
        return out


def full_bias(l, n=None):
    # expands the bias to the proper size. For convolutional layers, a full
    # output dimension of n must be specified.
    if isinstance(l, nn.Linear):
        return l.bias.view(1,-1)
    elif isinstance(l, nn.Conv2d):
        if n is None:
            raise ValueError("Need to pass n=<output dimension>")
        b = l.bias.unsqueeze(1).unsqueeze(2)
        if isinstance(n, int):
            k = int((n/(b.numel()))**0.5)
            return b.expand(1,b.numel(),k,k).contiguous().view(1,-1)
        else:
            return b.expand(1,*n)
    elif isinstance(l, Dense):
        return sum(full_bias(layer, n=n) for layer in l.Ws if layer is not None)
    elif isinstance(l, nn.Sequential) and len(l) == 0:
        return 0
    else:
        raise ValueError("Full bias can't be formed for given layer.")