import torch
from torchvision import models
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict
from get_args import train_args

in_arg = train_args()
network = in_arg.arch

def classifier(arch = network):
    if arch == "vgg":
        model = models.vgg11(pretrained=True)
        #freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
    
        classifier = nn.Sequential(OrderedDict([('0', nn.Linear(25088,in_arg.hidden_units[0])),
                                                ('1', nn.ReLU()),
                                                ('2', nn.Dropout(0.5)),
                                                ('3', nn.Linear(in_arg.hidden_units[0],in_arg.hidden_units[1])),
                                                ('4', nn.ReLU()),
                                                ('5', nn.Dropout(0.5)),
                                                ('6', nn.Linear(in_arg.hidden_units[1],102)),
                                                ('output',nn.LogSoftmax(dim=1))]))
        
        model.classifier = classifier
        
    elif arch == "resnet":
        model = models.resnet50(pretrained=True)
        #freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
    
        classifier = nn.Sequential(nn.Linear(2048,in_arg.hidden_units[0]),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(in_arg.hidden_units[0],in_arg.hidden_units[1]),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(in_arg.hidden_units[1], 102),
                                   nn.LogSoftmax(dim=1))
        
        model.fc = classifier
        
    return model