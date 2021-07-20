from fastai.vision import *
from fastai.metrics import error_rate
import sys
import copy

from order_metrics import *


###############################################
# Owa helpers

def get_parametric_owa(size, func=lambda x: x):
    weights = np.zeros(size)
    for i in range(1, size + 1):
        weights[i-1] = func(i / size) - func((i-1) / size)
    return weights

def get_parametric_owa_torch(size, func=lambda x: x):
    weights = torch.zeros(size, device=torch.device('cuda'))
    for i in range(1, size + 1):
        weights[i-1] = func(i / size) - func((i-1) / size)
    return weights

def soft_max_generator(x, a=1, b=0.2):
    if x < a:
        return np.power(x/a, b)
    else:
        return 1
    
def soft_max_generator_alt(x, a=60):
    return 1 - np.power(a, -x)

def average_generator(x):
    return x

def max_generator(x):
    if x == 0:
        return 0
    else:
        return 1
    
def two_param_generator(x, a1, a2):
    p1 = min(max(a1, 0), 1)
    p2 = min(max(a2, 0), 1)
    if p2 < p1:
        p1, p2 = p2, p1
    if x < p1:
        return 0
    elif p2 < x:
        return 1
    else:
        return (x - p1) / (p2 - p1)

def rim_generator(x, a=1):
    assert(a > 0)
    if a == 1:
        return x
    else:
        return (np.power(a, x)-1) / (a-1)
    return x

#Lookup table for RIM
rim_a_val = np.exp(np.arange(-100, 100, 0.05))
rim_orness = (rim_a_val - 1 - np.log(rim_a_val)) / ((rim_a_val-1) * np.log(rim_a_val))
rim_orness[np.isnan(rim_orness)] = 0.5

def rim_generator_for_orness(x, orness):
    idx = (np.abs(rim_orness - orness)).argmin()
    a = rim_a_val[idx]

    if a == 1:
        return x
    else:
        return (np.power(a, x)-1) / (a-1)
    return x
    
# Owa helpers
###############################################


###############################################
# Aggregation layer

class WeightConstraint():
    def __init__(self, mode):
        self.mode = mode
        if mode not in ['full_owa', 'free']:
            raise Exception(f'Unknown weight constraint mode {mode}')

    def apply(self, weight):
        if self.mode == 'full_owa':
            sm = F.leaky_relu(weight, 0.01, False)
            smsum = torch.sum(sm, dim=1)
            smsum = smsum.expand(1, smsum.shape[0]).T.expand(sm.shape)
            sm = sm / smsum
            sm = torch.where(torch.isnan(sm), torch.zeros_like(sm), sm)
            return sm
        elif self.mode == 'free':
            return weight
            

class Aggregation(Module):
    def __init__(self, ni, nf, initmethod=None, constrainmode='free', init_denominator=None):
        self.ni = ni
        self.nf = nf
        self.constrainmode = constrainmode
        self.constraint = WeightConstraint(constrainmode)
        if init_denominator is None:
            self.init_denominator = self.ni
        else:
            self.init_denominator = init_denominator
        if initmethod == None:
            self.weight = nn.Parameter(torch.rand(nf, ni) / self.init_denominator)
        elif initmethod == 'identity':
            w = torch.abs(torch.eye(nf, ni) + torch.randn(nf, ni)) / self.ni
            self.weight = nn.Parameter(w)
        elif initmethod == 'double':
            w = torch.abs(torch.eye(nf, ni) + torch.randn(nf, ni)) / self.ni
            self.weight = nn.Parameter(w)
            a = int(nf / 2)
            b = int(nf - a)
            self.weight.data = torch.cat([self.weight[:, :a], self.weight[:, a+b:], self.weight[:, a:a+b]], 1)
        elif initmethod == 'imagenette':
            self.weight = nn.Parameter(torch.abs(torch.randn(nf, ni)) / self.init_denominator)
    
    def extra_repr(self):
        return f'{self.ni}, {self.nf}, {self.constrainmode}'
    
    def forward(self, input):
        sm = self.constraint.apply(self.weight)
        permuted = input.permute(0,2,3,1)
        output = torch.matmul(permuted, sm.T).permute(0,3,1,2)
        output = output
        return output

class DummyAggregation(Module):
    def __init__(self, ni, nf, inverse=False):
        self.nf = nf
        self.inverse = False

    def forward(self, input):
        output = input[:, :self.nf, :, :]
        if self.inverse:
            output = input[:, -self.nf:, :, :]
        return output

def torch_binom(n, k):
    mask = n >= k
    n = mask * n
    k = mask * k
    a = math.lgamma(n + 1) - math.lgamma((n - k) + 1) - math.lgamma(k + 1)
    return math.exp(a) * mask

class BinomialAggregation(Module):
    def __init__(self, ni, nf):
        self.ni = ni
        self.nf = nf
        self.orness = nn.Parameter(torch.rand(nf))
    
    def extra_repr(self):
        return f'{self.ni}, {self.nf}'
    
    def forward(self, input):
        w = torch.zeros(self.nf, self.ni).to(input.get_device())
        for i in range(1, self.nf + 1):
            w[:, i - 1] = torch.pow(1 - self.orness, i-1) * torch.pow(self.orness, self.nf - i) * torch_binom(self.nf-1, i-1)
        permuted = input.permute(0,2,3,1)
        output = torch.matmul(permuted, w.T).permute(0,3,1,2)
        output = output
        return output

# Aggregation layer
###############################################

    
###############################################
# OWA layer

class OWAlayer(Module):
    def __init__(self, ni, nf, sorting_func, aggregate='linear', concat=True, **kwargs):
        self.sorting_func = sorting_func
        self.ni = ni
        self.nf = nf
        self.concat = concat
        if aggregate == 'trim':
            self.aggregation = DummyAggregation(ni, nf)
        elif aggregate == 'trim-inverse':
            self.aggregation = DummyAggregation(ni, nf, True)
        elif aggregate == 'binomial':
            self.aggregation = BinomialAggregation(ni, nf)
        else:
            self.aggregation = Aggregation(ni, nf, initmethod=aggregate, **kwargs)
            
    def extra_repr(self):
        return f'(sort): {self.sorting_func.__name__}, (concat): {self.concat}'

    def forward(self, input):
        sorted = self.aggregation(reorder_images_by_rank(input, self.sorting_func(input)))
        if self.concat:
            output = torch.cat((input, sorted), 1)
        else:
            output = sorted
        return output

# OWA layer
###############################################


###############################################
# Pixel OWA layer
    
class PixelOWAlayer(Module):
    def __init__(self, ni, nf, aggregate='linear'):
        self.ni = ni
        self.nf = nf
        if aggregate == 'linear':
            self.aggregation = Aggregation(ni, nf)
        elif aggregate == 'linear-identity':
            self.aggregation = Aggregation(ni, nf, initmethod='identity')
        elif aggregate == 'linear-double':
            self.aggregation = Aggregation(ni, nf, initmethod='double')
        elif aggregate == 'linear-imagenette':
            self.aggregation = Aggregation(ni, nf, initmethod='imagenette')
        elif aggregate == 'trim':
            self.aggregation = DummyAggregation(ni, nf)
        elif aggregate == 'trim-inverse':
            self.aggregation = DummyAggregation(ni, nf, True)

    def forward(self, input):
        inputsort, _ = torch.sort(input, axis=1)
        sorted = self.aggregation(inputsort)
        images_concat = torch.cat((input, sorted), 1)
        print(input)
        print(images_concat)
        return images_concat
    
# Pixel OWA layer
###############################################
