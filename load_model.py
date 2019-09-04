import torch
from torchvision import transforms, datasets, models
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict

from get_args import predict_args


p_arg = predict_args()


def load_model(filepath = p_arg.checkpoint):
    
    checkpoint = torch.load(filepath)  
    
    if checkpoint["arch"] == "vgg11":
        model = models.vgg11(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
    
        classifier = nn.Sequential(OrderedDict([('0', nn.Linear(checkpoint["input_size"], checkpoint["hidden_units_1"])),
                                                ('1', nn.ReLU()),
                                                ('2', nn.Dropout(0.5)),
                                                ('3', nn.Linear(checkpoint["hidden_units_1"],checkpoint["hidden_units_2"])),
                                                ('4', nn.ReLU()),
                                                ('5', nn.Dropout(0.5)),
                                                ('6', nn.Linear(checkpoint["hidden_units_2"],checkpoint["output_size"])),
                                                ('output',nn.LogSoftmax(dim=1))]))
        
        model.classifier = classifier
        
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        
        return model
    
    elif checkpoint["arch"] == "resnet50":
        model = models.resnet50(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
            
        classifier = nn.Sequential(nn.Linear(checkpoint["input_size"], checkpoint["hidden_units_1"]),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(checkpoint["hidden_units_1"],checkpoint["hidden_units_2"]),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(checkpoint["hidden_units_2"],checkpoint["output_size"]),
                                   nn.LogSoftmax(dim=1))
        
        model.fc = classifier
    
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
    
        return model

#print(load_model())