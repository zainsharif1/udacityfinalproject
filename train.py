#import the necessary PyTorch modules
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models

from get_args import train_args
from classifier import classifier
from data_load import data_load

in_arg = train_args()

#setup model
model = classifier()

# set the gradient descent function, learning rate, optimizer and gpu state
criterion=nn.NLLLoss()
learning_rate = in_arg.learning_rate

if in_arg.arch == "vgg":
    optimizer = optim.Adam(model.classifier.parameters(), lr =learning_rate)
elif in_arg.arch == "resnet":
    optimizer = optim.Adam(model.fc.parameters(), lr =learning_rate)
   
GPU = in_arg.gpu

#retrieve the image data
dataloaders= data_load()[0]
image_datasets = data_load()[1]

def model_train(model = model, error_function = criterion, optimizer = optimizer, dataloaders = dataloaders, GPU = GPU, image_datasets = image_datasets,                     save_dir = in_arg.save_dir):
    ''' The function trains a given model on a dataset of training images, validates the accuracy on a validation set
        and then saves the model architecture into a provided save directory
    
    Inputs:
    
        - model : the model to be trained can be vgg13 (default) or resnet50 if entered as an argument with architecture determined
                  through the classifier function
        - error_function : gradient descent function
        - optimizer : optimizer function
        - dataloaders :  defined by the image data sets and transforms in the data_load function
        - GPU : Boolean state of GPU obtained from argparse
        - image_datasets : image data sets from data_load function
        - save_dir : directory to save model architechture to
    
    Returns:
    
        Prints out training loss, validation loss and accuracy for number of epochs specified
        
        Returns trained model
    '''
    if GPU == False:
        power = "cpu"
    elif GPU == True:
        power = "cuda"
        
    model.to(power)

    epochs =in_arg.epochs


    for epoch in range(epochs):
        train_loss = 0
        for images, labels in dataloaders["train_loader"]:
            images, labels = images.to(power), labels.to(power)
        
            optimizer.zero_grad()
        
            logps = model(images)
            loss = error_function(logps, labels)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
       
        else:
            accuracy = 0
            validation_loss = 0
        
            with torch.no_grad():
                model.eval()
            
                for images, labels in dataloaders["validation_loader"]:
                    images, labels = images.to(power), labels.to(power)
                
                    logps = model(images)
                    loss = criterion(logps, labels)
                    validation_loss += loss.item()
                
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
            print("Running epoch {}".format(str(epoch+1)))
            print("Training loss is: {}".format(str(train_loss/len(dataloaders["train_loader"]))))
            print("Validation loss is: {}".format(str(validation_loss/len(dataloaders["validation_loader"]))))
            print("Accuracy: {}%\n".format(str(accuracy/len(dataloaders["validation_loader"])*100)))
        
            train_loss = 0
            model.train()
            

    model.class_to_idx = image_datasets["train_data"].class_to_idx
    
    if in_arg.arch == "vgg":
        checkpoint = { "arch" : "vgg11", "state_dict" : model.state_dict(), "class_to_idx" : model.class_to_idx,
                      "input_size" : 25088, "hidden_units_1": in_arg.hidden_units[0], "hidden_units_2" : in_arg.hidden_units[1], "output_size" : 102 }
    
    elif in_arg.arch == "resnet":
        checkpoint = { "arch" : "resnet50", "state_dict" : model.state_dict(), "class_to_idx" : model.class_to_idx,
                      "input_size" : 2048, "hidden_units_1": in_arg.hidden_units[0], "hidden_units_2" : in_arg.hidden_units[1], "output_size" : 102 }
        
    torch.save(checkpoint, save_dir)
    
    print("PyTorch model data saved to: {}".format(save_dir))
    
    return model

model_train()