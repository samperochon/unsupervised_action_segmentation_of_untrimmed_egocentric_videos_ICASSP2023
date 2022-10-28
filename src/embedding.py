import sys
import os
import time
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


from .const import *
from .deep_features import DeepFeatures
from .utils import fi

class Embedding(object):
    def __init__(self, 
                reduce_embedding=DEFAULT_REDUCE_EMBEDDING,
                dim_embedding=DEFAULT_EMBEDDING_DIMENSION_REDUCTION,
                config=DEFAULT_CONFIG,
                verbosity=VERBOSITY):
                
        self._extractor = DeepFeatures(model_name=config['model_name'],
                                            config=config,
                                            verbosity=config['verbosity'] )

        self.embedding = None
        self._embedding = None
        self.reduce_embedding = reduce_embedding
        self.dim_embedding = dim_embedding

        self.gram = None

        # Maintenance related
        self.verbosity = verbosity
        self.config = config

    def extract(self, dataset, idx_to_extract, path=None, *args, **kwargs):
        
        if path is not None:
            
            assert os.path.isfile(path), "Invalid feature vectors path: {}".format(path)
            self.embedding = np.load(path)
            
        else:

            self.embedding = self._extractor.extract(dataset, idx_to_extract, *args, **kwargs)
            
        self._embedding = deepcopy(self.embedding)


        print("Extraction done") if self.verbosity > 2 else None

        if self.reduce_embedding:

            self.reduce()

        self.dim_embedding = self.embedding.shape[0]

        print("Reducing done") if self.verbosity > 2 else None

        return 

    def compute_gram(self, verbose=False, *args, **kwargs):

        self._gram = self.embedding.T@self.embedding
        self.gram = deepcopy(self._gram)
        self.gram[self.gram <=0] = 0
        self.gram[self.gram >=1] = 1

        return self.gram

    def reduce(self, dim_embedding=None):

        # Change only here the dim o the embedding ahead of computation, otherwise it is always the dimension of the embedding.
        if dim_embedding is not None:
            
            #if False and dim_embedding <  self.embedding.shape[1] or dim_embedding > self.dim_embedding:
            if dim_embedding > np.min((self.dim_embedding, self.embedding.shape[1])):
                
                print("Aborted. Dimension should be higher than the number of samples ({}) and lower than the original embedding dimension ({}).".format(self.embedding.shape[1], self.dim_embedding))
                return 
            pca = PCA(n_components=dim_embedding)
            self.embedding = pca.fit_transform(self.embedding.transpose()).T
            self.dim_embedding = dim_embedding

        else:
            self.dim_embedding = self._embedding.shape[0]
            self.embedding = deepcopy(self._embedding)


        return 
    