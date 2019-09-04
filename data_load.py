import torch
from torchvision import transforms, datasets, models
from get_args import train_args



in_arg = train_args()

def data_load():
    
    #set data directories
    data_dir = in_arg.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Define transforms for the training, validation, and testing sets
    data_transforms = {"train_transforms": transforms.Compose([transforms.RandomRotation(40),
                                                               transforms.RandomResizedCrop(224),
                                                               transforms.RandomHorizontalFlip(),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                        "validation_transforms": transforms.Compose([transforms.Resize(224),
                                                                     transforms.CenterCrop(224), 
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                            "testing_transforms":transforms.Compose([transforms.Resize(224),
                                                                     transforms.CenterCrop(224),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # Load the datasets with ImageFolder
    image_datasets = {"train_data":datasets.ImageFolder(train_dir, transform=data_transforms["train_transforms"]),
                       "validation_data":datasets.ImageFolder(valid_dir, transform=data_transforms["validation_transforms"]),
                       "test_data":datasets.ImageFolder(test_dir,transform=data_transforms["testing_transforms"])}

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {"train_loader":torch.utils.data.DataLoader(image_datasets["train_data"], batch_size=64, shuffle=True),
                   "validation_loader":torch.utils.data.DataLoader(image_datasets["validation_data"], batch_size=64),
                   "test_loader":torch.utils.data.DataLoader(image_datasets["test_data"], batch_size=64, shuffle=True)}
    
    return [dataloaders, image_datasets]