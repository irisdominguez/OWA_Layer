from fastai.vision import *
import time
import random
import json
from model import *
from utils import *

def load_dataset(dataset_name, bs):
    if dataset_name == "cifar10":
        path = untar_data(URLs.CIFAR, fname='./data/cifar10.tgz', dest='./data')
        tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
        target_size = 32
        valid = "test"
    elif dataset_name == "cifar100":
        path = untar_data(URLs.CIFAR_100, fname='./data/cifar100.tgz', dest='./data')
        tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
        target_size = 32
        valid = "test"
    elif dataset_name == "imagenette":
        path = untar_data(URLs.IMAGENETTE, fname='./data/imagenette2.tgz', dest='./data')
        tfms = ([*rand_pad(32, 256), flip_lr(p=0.5)], [])
        target_size = 256
        valid = "val"
        
    il = ImageList.from_folder(path)
    sd = il.split_by_folder(train='train', valid=valid)
    ll = sd.label_from_folder()
    ll = ll.transform(tfms, size=target_size)
    data = ll.databunch(bs=bs).normalize()
    return data

def getConvBlock(inc, outc):
    return [nn.Conv2d(inc, outc, kernel_size=3, padding=1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)]

def create_large_model(data):
    # Puntos de inserción para esta red:
    # Bloque 1: 3 32x32x64,
    # Bloque 2: 8 16x16x64, 12 16x16x128,
    # Bloque 3: 17 8x8x128, 21 8x8x256, 
    # Bloque 4: 26 4x4x256, 30 4x4x512,
    # Bloque 5: 35 2x2x512, 39 2x2x512,
    
    s = 8
    
    n1 = 64
    n2 = 128
    n3 = 256
    n4 = 512
    n5 = 512
    
    model = nn.Sequential(
        *getConvBlock(3, n1),
        nn.Identity(),
        *getConvBlock(n1, n1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Identity(),
        *getConvBlock(n1, n2),
        nn.Identity(),
        *getConvBlock(n2, n2),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Identity(),
        *getConvBlock(n2, n3),      
        nn.Identity(),
        *getConvBlock(n3, n3),      
        nn.Identity(),
        *getConvBlock(n3, n3),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Identity(),
        *getConvBlock(n3, n4),
        nn.Identity(),
        *getConvBlock(n4, n4),
        nn.Identity(),
        *getConvBlock(n4, n4),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Identity(),
        *getConvBlock(n4, n5),
        nn.Identity(),
        *getConvBlock(n5, n5),
        nn.Identity(),
        *getConvBlock(n5, n5),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        
        nn.Linear(n5 * s * s, 4096),
        nn.Linear(4096, 4096),
        nn.Linear(4096, len(data.y.classes))
    )

    return model


def create_small_model(data):
    # Puntos de inserción para esta red:
    # Bloque 1: 3 32x32x64
    
    x = data.x[0].shape[1]
    y = data.x[0].shape[2]

    n1 = 64
    
    model = nn.Sequential(
        *getConvBlock(3, n1),
        nn.Identity(),
        *getConvBlock(n1, n1),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        
        nn.Linear(n1 * (x // 2) * (y // 2), len(data.y.classes)),
        nn.Softmax(dim=1)
    )

    return model


def add_layer(learn, layer, pos, nf, **kwargs):
    with torch.no_grad():
        old = learn.model[pos + 1]
        learn.model[pos] = layer(old.in_channels, nf, **kwargs)
        learn.model[pos + 1] = nn.Conv2d(old.in_channels + nf, old.out_channels, kernel_size=3, padding=1)
        learn.model.to(torch.cuda.current_device())
    return learn

def mod_learner_owa(learn, pos, order, aggregate, constrainmode, init_denominator):
    for p in pos:
        add_layer(
            learn,
            OWAlayer,
            p,
            pos[p],
            sorting_func = order,
            aggregate = aggregate,
            constrainmode = constrainmode,
            init_denominator = init_denominator)
    return learn

def mod_learner_pixel_owa(learn, pos, aggregate):
    for p in pos:
        add_layer(
            learn,
            PixelOWAlayer,
            p,
            pos[p],
            aggregate = aggregate)
    return learn

@dataclass
class SilenceRecorder(Callback):
    learn:Learner
    def __post_init__(self):
        self.learn.recorder.silent = True
    
def reset_optimizer(learn):
    learn = Learner(learn.data, 
            learn.model, 
            metrics=learn.metrics,
            callback_fns=[SilenceRecorder])
    return learn

def print_OWA_weights(model, itid, path):
    for i, l in enumerate(model):
        if type(l) == OWAlayer:
            if type(l.aggregation) == Aggregation:
                w = l.aggregation.weight.data.cpu().numpy().astype('float64')
                
                sort_name = l.sorting_func.__name__.replace('_image_batch','')
                
                print(w)
                
                mkdir_p(path)
                np.save(os.path.join(path, f'{itid}_{i}'), w)

def log_results(results, log_file):
    class JSONEncoder(json.JSONEncoder):
        def default(self, obj):
            return str(obj)
    f = open(log_file,'a+')
    f.write(JSONEncoder().encode(results))
    f.write('\n')
    f.close()
    
def lr_finder(config):
    model = config['model']
    data = config['data']
    modelobj = model(data)
    learn = Learner(data, modelobj, metrics=accuracy)
    learn.lr_find()
    learn.recorder.plot()

def run_config(config):
    pos = 'reference'
    order_name = 'no order'
    aggregate = 'no aggregation'
    lr = config['lr']
    model = config['model']
    epochs = config['epochs']
    its = config['its']
    data = config['data']
    matrixpath = config['matrixpath']
    metric = accuracy
    if 'metric' in config:
        metric = config['metric']
    
    if config['type'] == 'reference':
        mod_learner = lambda x: x
        
    elif config['type'] == 'layer':
        pos = config['pos']
        order = config['order']
        aggregate = config['aggregate']
        constrainmode = config['constrainmode']
        if 'init_denominator' in config:
            init_denominator = config['init_denominator']
        else:
            init_denominator = None
        order_name = order.__name__.replace('_image_batch','')
        mod_learner = partial(mod_learner_owa, pos=pos, order=order, aggregate=aggregate, 
                              constrainmode=constrainmode, init_denominator=init_denominator)

    elif config['type'] == 'pixel':
        pos = config['pos']
        aggregate = config['aggregate']
        order_name = 'pixel'
        mod_learner = partial(mod_learner_pixel_owa, pos=pos, aggregate=aggregate)

    results = []

    for i in range(its):
        itid = f'{random.getrandbits(64):016x}'
        print(f'Iteration {i + 1} of {its}')
        modelobj = model(data)
        learn = Learner(data, modelobj, metrics=metric)
        learn = mod_learner(learn)
        learn = reset_optimizer(learn)
        
        if i == 0:
            print(learn.model)

        start = time.time()
        learn.fit_one_cycle(epochs, lr)
        #learn.fit(epochs, lr)
        end = time.time()
        acc = torch.mean(torch.tensor(learn.recorder.metrics[-1]))
        
        print_OWA_weights(learn.model, itid, matrixpath)
        
        results.append({
            'id': itid,
            'accuracy': list(np.array(learn.recorder.metrics)[:, 0].astype(float)),
            'time': end-start
        })
        
    returndict = {
        'config': config,
        'results': results
    }
    return returndict