import argparse


def train_args():
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data_dir", type=str, default= "flowers", help = "path to the folder of flower images")
    
    parser.add_argument("--save_dir", type=str, default = "vgg_saved_model.pth", help = "path to the folder where the checkpoint is saved")
    
    parser.add_argument("--arch", type=str, default = "vgg", help = "architecture of model")
    
    parser.add_argument("--learning_rate", type=float, default = 0.001, help = "learning rate to be applied for gradient descent")
    
    parser.add_argument("--hidden_units", action = 'append', default = [4096,1000], help = "provide number of units for 2 hidden layers")
    
    parser.add_argument("--epochs", type=int, default = 10, help = "number of epochs/iterations for the training model")
    
    parser.add_argument("--gpu", action = "store_true", default=False, help = "enable gpu mode when entered as an argument to command line")
    
    
    return parser.parse_args()

def predict_args():
    
    parser1 = argparse.ArgumentParser()

    parser1.add_argument("input", type=str, default="flowers/test/1/image_06743.jpg", help = "path to image to be used for processing image")
    
    parser1.add_argument("checkpoint", type=str, default = "checkpoint.pth", help = "path from which the checkpoint is loaded")
    
    parser1.add_argument("--top_k", type = int, default = 5, help = "top K classes of probabilities")
    
    parser1.add_argument("--category_names", type = str, default = "cat_to_name.json", help = "mapping of categories to real names")
    
    parser1.add_argument("--gpu", action = "store_true", default=False, help = "enable gpu mode when entered as an argument to command line")
    
    return parser1.parse_args()