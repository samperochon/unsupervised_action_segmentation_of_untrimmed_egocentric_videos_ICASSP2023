import sys
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import cv2

import ruptures as rpt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from skimage.filters import threshold_otsu

from .const import DEFAULT_CONFIG, MIN_SEGMENTS_SIZE
from .utils import (fi, pairwise,  join_and_discard)

from .embedding import Embedding
from .metrics import Annotation, f_score


class Pipeline(object):
    def __init__(self, 
                dataset, 
                annotation=None,
                output_path = './outputs',
                config=DEFAULT_CONFIG,
                verbosity=DEFAULT_CONFIG['verbosity']):

        # Dataset related
        self.dataset = dataset
        self.annotation = annotation if annotation is not None else Annotation()
        self.tau_sample = config['tau_sample']
        self.idx_embedding = np.arange(0, self.dataset.n_frames, self.tau_sample)
        
        # Embedding related
        self._embedding = Embedding(dim_embedding = config['dim_embedding'],
                                    config=config,
                                    verbosity=config['verbosity'])

        # Rupture detection 
        self.penalty = None
        self.rupture_detector = None
        self.ruptures_on = None
        self.cpt = None
        self.cpt_frame = None

        # Clustering related
        self.n_clusters = None 

        self.pred_frames_label = None
        self.embedding_label = np.array([np.nan] * len(self.idx_embedding))
        self.segments_label = None

        # Outliers related
        # All outlier detection method should return a binary mask, with 1 for outliers, with the same length as the idx_embedding (using the embedding referential, not the video one).
        self.remove_outliers = config['remove_outliers']
        self._outlier_methods = config['outlier_methods']
        self.outlier_methods_dict = {'custom': self._find_outliers_custom,
                                    'silhouette_score': self._find_outliers_frame_silhouette}
                                    
        self._mask_outliers = np.zeros((len(self.outlier_methods), len(self.idx_embedding)), dtype=np.uint8) 
        self.mask_outliers = np.zeros(len(self.idx_embedding), dtype=np.uint8)
        self.idx_outliers = []
        self.idx_outliers_segments = []
        self.outliers_removed = False


        # Maintenance related
        self.verbosity = verbosity
        self.config = config
        
        os.makedirs(output_path, exist_ok=True)
        self.output_path = output_path

    @property
    def outlier_methods(self):
        return self._outlier_methods

    @outlier_methods.setter
    def outlier_methods(self, value):
        self._outlier_methods = value
        self._mask_outliers = np.zeros((len(self.outlier_methods), len(self.idx_embedding)))
        self.mask_outliers = np.zeros(len(self.idx_embedding))
        self.outliers_removed = False


    @property
    def gram(self):
        return self._embedding.gram 

    @property
    def embedding(self):
        if hasattr(self.dataset, 'embedding'):
            return self.dataset.embedding
        else:
            return self._embedding.embedding

    def compute_embedding(self, path=None, force=False, save=False, *args, **kwargs):
        
        self._embedding.extract(self.dataset, idx_to_extract=self.idx_embedding, path=path, *args, **kwargs)

        self._embedding.dim_embedding = int(self._embedding.embedding.shape[0])
        self.config['dim_embedding'] = int(self._embedding.embedding.shape[0])

        # Compute the gram. 
        self.compute_gram()

        return 


    def compute_gram(self, gram_post_process=None, *args, **kwargs):

        if self._embedding.embedding is not None:
            self._embedding.compute_gram(*args, **kwargs)

        return 

    def reduce(self, dim_embedding=None):

        if self.embedding is not None:
            self._embedding.reduce(dim_embedding)

        return
    
    def remove_outliers_method(self, verbose=False):

        if len(self.outlier_methods)==0:
            print("No outlier method selected.")
            return

        for i, method in enumerate(self.outlier_methods):

            if method == 'silhouette_score': 
                continue

            self._mask_outliers[i, :] = self.outlier_methods_dict[method](verbose=verbose)

        if len(self.outlier_methods) >1 :

            self.mask_outliers[np.argwhere(self._mask_outliers.sum(axis=0)>0).squeeze()] = 1
    
        else:

            self.mask_outliers = self._mask_outliers.squeeze()

        self.idx_outliers = np.argwhere(self.mask_outliers==1).squeeze()
        self.outliers_removed = True
        return 


    def detect_ruptures(self, penalty=None, ruptures_on = 'merge_ts', remove_outliers = True, verbose=False):

        """
            Compute self.cpt and self.cpt_frame.

            Add those to the annotation class. 

            One can then call self.compute_segments_representation() to compute self.segments_embedding.

        """

        self.ruptures_on = ruptures_on

        gram = self._embedding.gram
        embedding = self._embedding.embedding.T


        if self.outliers_removed and remove_outliers :
            embedding = embedding[self.mask_outliers==0]

            gram = embedding @ embedding.T
            
        # Set penalty parameter
        if penalty is None:
            self.penalty = DEFAULT_PENALTY
        else:
            self.penalty = penalty
        
        penalty = self.penalty/np.log(embedding.shape[0])

        # Set minimum segment length
        min_size = int(self.dataset.fps*MIN_SEGMENTS_SIZE/self.tau_sample)

        # Z-transform the multivariate signal
        embedding = (embedding  - np.mean(embedding , axis=1)[:, np.newaxis])/ np.var(embedding , axis=1)[:, np.newaxis]

        # On the embedding
        self.rupture_detector = rpt.KernelCPD(kernel="cosine", min_size=min_size)
        self.cpt = self.rupture_detector.fit_predict(embedding, pen=penalty)

        if self.outliers_removed and remove_outliers:
            from_to_dict = {}
            count = 0
            for idx in range(len(self.idx_embedding)):
                if idx in self.idx_outliers:
                    from_to_dict[idx-count] = idx
                    count +=1
                else:
                    from_to_dict[idx-count] = idx
            from_to_dict[gram.shape[0]] = len(self.idx_embedding)
                    
            self.cpt = [from_to_dict[c] for c in self.cpt]

        print("Done. Detected {} ruptures.".format(len(self.cpt)-2)) if self.verbosity > 2 else None

        # Add ruptures of the outliers
        outliers_cpt = list(np.argwhere(np.abs(np.ediff1d(self.mask_outliers))>0).flatten())
        self.cpt  = sorted(list(set(self.cpt + outliers_cpt)))
        print("Total {} ruptures.".format(len(self.cpt)-2)) if self.verbosity > 2 else None
        
        # Remove segments of size less than half a secont (artifact from the removal of outliers frames)
        min_size=np.ceil(self.dataset.fps*.2/self.tau_sample).astype(int)
        self.cpt =  [j for i, j in pairwise(self.cpt) if j-i > min_size]
        print("Total {} ruptures after pruning".format(len(self.cpt)-2)) if self.verbosity > 2 else None
        
        self.mask_outliers[self.cpt[:-1]] = 1
        self.idx_outliers = np.argwhere(self.mask_outliers==1).squeeze()
        self.cpt = [0] + self.cpt


        # Compute action F1 score
        if self.annotation.gt_label is not None:

            predicted_actions = (self.mask_outliers==0).astype(int)

            gt_label = deepcopy(self.annotation.gt_label)
            gt_label[np.argwhere(np.abs(np.ediff1d(gt_label))>0).flatten()] = 0
            gt_actions = (gt_label!=0).astype(int)

            self.f1_actions = self._compute_segmental_f1(predicted_actions,  gt_actions, overlap=.25, verbose=False)



        #print("Done. Detected {} ruptures.".format(len(self.cpt)-2)) if self.verbosity > 2 else None

        # Init segments labels
        self.segments_label = np.array([np.nan]*(len(self.cpt)-1))
        self.compute_cpt_frame(tau=self.tau_sample)

        return   

    def compute_cpt_frame(self, tau=None):

        if tau is None:
            tau=self.tau_sample

        self.cpt_frame = (np.array(self.cpt)*tau).astype(int)

        if self.cpt_frame[-1] > self.dataset.n_frames:
            self.cpt_frame[-1] = self.dataset.n_frames
        
        return

    def cluster_frames(self, method='kmeans', n_clusters=None, verbose=True, *args, **kwargs):

        """
            Takes n_clusters as input (optionnal), and use the self.segments_embedding to cluster each of the segment. 

            output:

            self.segments_label

        """
        
        # Define number of cluster
        self.n_clusters = n_clusters 

        # Init. the vectors used
        
        embedding = self._embedding.embedding.T

        vectors = embedding[self.mask_outliers==0]


        # Cluster the representations using K-Means 
        kmeans = KMeans(n_clusters=self.n_clusters, algorithm='auto', init='k-means++', random_state=0).fit(vectors)
        labels = kmeans.labels_+1

                
        # Assign the labels of each inlier vectors
        idx_reading_value = 0
        for idx in range(self.embedding.shape[1]):
            if idx in self.idx_outliers:
                self.embedding_label[idx] = 0
            else:
                self.embedding_label[idx] = labels[idx_reading_value]
                idx_reading_value+=1

        self.embedding_label = self.embedding_label.astype(int)
        
        for segment_index, (i, j) in enumerate(pairwise(self.cpt)):
            self.segments_label[segment_index] = np.bincount(self.embedding_label[i:j]).argmax()
        
        # Unlabelled background segments if isolated between two same activities (if length is less than 2s)
        min_size=np.ceil(self.dataset.fps*.25/self.tau_sample).astype(int)
        segments_length = [j-i for i, j in pairwise(self.cpt)]
        previous_label = self.segments_label[0]
        for i, segment_label in enumerate(self.segments_label[1:-1]):
            next_label = self.segments_label[i+2]
            
            #if segment_label==0 and next_label==previous_label!=0 and segments_length[i+1]<=min_size:

            if next_label==previous_label!=0 and segments_length[i+1]<=min_size:
                self.segments_label[i+1] = next_label

            previous_label = segment_label

        self._propagate_labels()
        #self._find_outlier_segments(verbose=True)

        if 'silhouette_score' in self.outlier_methods:
            self._find_outliers_frame_silhouette(verbose=verbose)

        self._segments_outliers_removal(verbose=verbose)        

        return 


    def create_outputs(self):
        results = pd.DataFrame({'n_frames': self.dataset.n_frames, 
                                'acc': self.annotation.accuracy, 
                                'frame_f1_macro': self.annotation.f1_macro, 
                                'jaccard': self.annotation.iou, 
                                'edit': self.annotation.edit, 
                                'f1_10': self.annotation.f1_10, 
                                'f1_25': self.annotation.f1_25, 
                                'f1_50': self.annotation.f1_50}, index=[0])

        results.to_csv(os.path.join(self.output_path, 'performances.csv'), index=False)

        predicted_actions = pd.DataFrame({'gt_label': self.annotation.gt_label, 
                                        'pred_label' :  self.annotation.pred_frames_label})

        predicted_actions.to_csv(os.path.join(self.output_path, 'gt_and_pred_labels.csv'), index=False)

        return 



    def _segments_outliers_removal(self, threshold_outliers=.5, verbose=True):

        # Perform the union of both outliers removal steps
        if len(self.outlier_methods) >1 :
    
            self.mask_outliers = np.zeros(len(self.idx_embedding), dtype=np.uint8)
            self.mask_outliers[np.argwhere(self._mask_outliers.sum(axis=0)>0).squeeze()] = 1

        else:

            self.mask_outliers = self._mask_outliers.squeeze()    

        # Apply join and discard filtering
        join_size = np.ceil(self.dataset.fps*.3/self.tau_sample).astype(int)
        discard_size = np.floor(self.dataset.fps*.15/self.tau_sample).astype(int)
        self.mask_outliers = join_and_discard(deepcopy(self.mask_outliers), join_len=join_size, discard_len=discard_size, binary_mask=True).astype(int)

        for segment_index, (i, j) in enumerate(pairwise(self.cpt)):
            
            if np.mean(self.mask_outliers[i:j]) >= .5:
                self.segments_label[segment_index] = 0     
        
        self._propagate_labels()

        return 

    def _propagate_labels(self):

        """
        This function takes as input the detected ruptures (cpt_frame) in the video reference, the label of each segments (including outliers ones), the sampling rate (e.g. 5), and the number of frames of the initial video. 
        It computes the per frame label, and per embedding labels
        """

        embedding_label = np.zeros(len(self.idx_embedding))
        counter_segments = 0

        for i in range(0, len(self.idx_embedding)):

            # Is there a chnage of segment ? 
            if i == self.cpt[1:][counter_segments]:
                counter_segments+=1    

            embedding_label[i] = self.segments_label[counter_segments]
            
        self.embedding_label = embedding_label

        pred_frames_label = np.zeros(self.dataset.n_frames)
        counter_segments = 0

        for i in range(0, self.dataset.n_frames):

            # Is there a chnage of segment ? 
            if i == self.cpt_frame[1:][counter_segments]:
                counter_segments+=1    

            pred_frames_label[i] = self.segments_label[counter_segments]
            
        self.pred_frames_label = pred_frames_label.astype(int)

        self.annotation.add(pred_frames_label=self.pred_frames_label)
        
        
        return

    def idx2time(self, idx):
        #idx should be in the reference of the entire video, ie between 0 and self.dataset.n_frames
        if int(idx/self.dataset.fps//60)==0:
            return('{} | {:.2f}s'.format(idx//self.tau_sample, idx/self.dataset.fps%60))
        else:
            return('{} | {}m {:.2f}s'.format(idx//self.tau_sample, int(idx/self.dataset.fps//60), idx/self.dataset.fps%60))
    
    def time2idx(self, time):
        n_frames = int(time*self.dataset.fps)
        return n_frames//self.tau_sample


    def _compute_segmental_f1(self, frame_pred, frame_gt, overlap = .25, verbose=False):

    
        tp, fp, fn = f_score(frame_pred, frame_gt, overlap=overlap)

        precision = tp / float(tp + fp)
        recall = tp / float(tp + fn)

        try:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        except:
            f1=0

        f1 = np.nan_to_num(f1) 
        return f1
   

    def _find_outliers_custom(self, threshold=.75, verbose=False):

        if self.gram is None:
            self.compute_gram()

                    
        min_size = np.ceil(self.dataset.fps*MIN_SEGMENTS_SIZE/self.tau_sample).astype(int)
        delta =  np.ceil(self.dataset.fps*.1/self.tau_sample).astype(int)

        array = np.zeros(self.gram.shape[0])

        for i in range(self.gram.shape[0]):

            array[i]  = np.max([np.mean(self.gram[i, max(0, i-min_size-delta):i-min_size+delta+1]), np.nanmean(self.gram[i, min(self.gram.shape[0]-1, i+min_size-delta): min(self.gram.shape[0]-1, i+min_size+delta+1)])]) 


        self.custom_array = array
        mask = (np.array(array) <= threshold).astype(int)


        # Apply join and discard filtering
        join_size = np.ceil(self.dataset.fps*.3/self.tau_sample).astype(int)
        discard_size = np.floor(self.dataset.fps*.15/self.tau_sample).astype(int)
        mask = join_and_discard(mask, join_len=join_size, discard_len=discard_size, binary_mask=True).astype(int)

        return mask

    def _find_outliers_frame_silhouette(self, verbose=False):
        """
            Compute silhouette score for every 
        
        """

        if len(np.unique(self.embedding_label)) == 1:
            return

        embedding = self.embedding
        label = self.embedding_label

        silhouette_samples_values = silhouette_samples(embedding.T, label,  metric='cosine')

        threshold = np.mean(silhouette_samples_values) - np.std(silhouette_samples_values)


        mask_outliers_silhouette = np.where((silhouette_samples_values <=threshold) | (silhouette_samples_values>.99))[0]

        
        mask_outliers = np.zeros(len(self.idx_embedding))
        idx_array = np.arange(len(self.idx_embedding))
        #mask_outliers[idx_array[np.argwhere(self.mask_outliers==0).flatten()][mask_outliers_silhouette]] = 1
        mask_outliers[mask_outliers_silhouette] = 1
        self._mask_outliers[-1, :] = mask_outliers

        return 
