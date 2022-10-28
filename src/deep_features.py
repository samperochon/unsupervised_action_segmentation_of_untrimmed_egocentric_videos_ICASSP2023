import os
import sys
import numpy as np
from tqdm import tqdm


import torch
#import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


from .const import * 

class DeepFeatures(object):
    """
        This class aims at generating a class of frame embedding: the Fisher Vectors. It basically takes as input a dataset  with a list of frames index to extract, 
        and return an array (the embedding) of those frames. 
        It uses the class DescriptorExtraction to extract descriptors from frames, but could use any other type of collection of object take from a frame. 

    """
    
    def __init__(self, 
                model_name=DEFAULT_MODEL_NAME,
                config=DEFAULT_CONFIG,
                verbosity = VERBOSITY):
        
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.model_name == 'resnet152':
            self.model =  torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True).eval()

        elif self.model_name == 'resnet50':
            self.model =  torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).eval()

        elif self.model_name =='inception_v3':
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).eval()

        elif self.model_name == 'vgg':
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True).eval()


        self.model.to(self.device)

        if self.model_name in ['resnet152', 'vgg', 'resnet50']:
            self.preprocess = transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                ])
        elif self.model_name in ['inception_v3']:
            self.preprocess = transforms.Compose([transforms.Resize(299),
                                                    transforms.CenterCrop(299),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                ])


        # Maintenance related
        self.verbosity = verbosity
        self.config = config

    def extract(self, dataset, idx_to_extract, *args, **kwargs):
        embeddings = None

        for idx in tqdm(idx_to_extract):
            input_image = Image.fromarray(dataset[idx])
            input_tensor = self.preprocess(input_image).unsqueeze(0).to(self.device)

            if embeddings is None:
                with torch.no_grad():
                    embeddings = self.model(input_tensor)
                
            else:
                with torch.no_grad():
                    embedding = self.model(input_tensor)
                embeddings = torch.cat((embeddings, embedding), 0)
                
        embeddings = embeddings.cpu().detach().numpy().transpose()

        embeddings /= np.linalg.norm(embeddings, axis=0)

        return embeddings
    
