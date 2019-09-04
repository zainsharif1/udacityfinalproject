from PIL import Image
import numpy as np
from get_args import predict_args

p_arg = predict_args()

def process_image(image = p_arg.input):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    
    width, height = im.width, im.height
    aspect = width/height
    
    if width != height and width < height:
        im_resized = im.resize((256,int(256/aspect)))
    elif width != height and height<width:
        im_resized = im.resize((int(256*aspect),256))
    
    im_crop = im_resized.crop(((im_resized.width-224)/2,(im_resized.height-224)/2,(im_resized.width+224)/2,(im_resized.height+224)/2))
     
    np_image = np.array(im_crop)/255
    
    means = np.array([0.485,0.456,0.406])
    stdevs= np.array([0.229,0.224,0.225])
    
    np_normalized = (np_image - means)/stdevs
       
    return np.transpose(np_normalized,(2,0,1))