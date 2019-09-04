import torch
import numpy as np
from torchvision import transforms, datasets, models
import json

from get_args import predict_args
from load_model import load_model
from process_image import process_image

p_arg = predict_args()

def predict(image_path = p_arg.input, model = load_model(p_arg.checkpoint), GPU = p_arg.gpu, top_k = p_arg.top_k ):
    '''This function iterates through images in a specified loader and provides the top K classes of flowers by probability
    as determined by the model.
    
    Inputs:
    
       - image_path: path of image for which the model is going to predict the class
       - model : loaded model from a checkpoint to test
       - GPU : Boolean state of GPU obtained from argparse
       - top_k : top K probabilities to display obtained from argparse
    
    Returns: 
        Top K classes of flowers for the associated image
    
    
    '''
    
    # Evaluate model on images in the dataloader and return the top 5 probabilities 
    #along with classes and their respective names
    
    with open(p_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    if GPU == False:
        power = "cpu"
    elif GPU == True:
        power = "cuda"
    
    model.to(power)
    model.eval()
    
    img = torch.from_numpy(process_image(image_path)).float().to(power).unsqueeze_(0)

    with torch.no_grad():
        output = model.forward(img)
    
    ps = torch.exp(output)
    
    top_p, top_class = ps.topk(top_k, dim=1)
    
    names=list()
    for c in top_class[0].tolist():
        for k, v in model.class_to_idx.items():
            if c == v:
                names.append(cat_to_name[k])
                
    return np.around(top_p[0].cpu().numpy(), decimals =3), names

print(predict())