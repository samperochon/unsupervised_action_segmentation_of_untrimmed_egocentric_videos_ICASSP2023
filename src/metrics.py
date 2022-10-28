import sys
import os
import json
from copy import deepcopy
from  glob import glob 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from sklearn import metrics


from .const import *
from .utils import decompose

from sklearn.metrics import jaccard_score, f1_score
overlaps = [.1, .25, .5]

class Annotation(object):
    """
        Class handling all the annotations, whether there are grund truth or predicted.
    Time reference is always the one of the video (no sampling). 
    The start frames and stop frames always contains the video first and last frames.
    """
    
    def __init__(self,output_path = './outputs',names = ['Ground Truth', 'Prediction (ours)'],overlap_list=[.1, .25, .5]):
        """ 
            We assume here that if the cpt are provided, they contain a [0] and [n_frame] values.
            If we provide start and end frames, we add [0] and [n_frames]
        """

        self.gt_available = False
        self.prediction_is_available = True 
        self.pred_frames_label = None
        self.gt_label = None

        
        self.names = names
        self.overlap_list = overlap_list
        
        # Segment-wise metrics
        self.f1_10 = None
        self.f1_25 = None
        self.f1_50 = None
        
        
        # Frame-wise metrics
        self.accuracy = None
        self.f1_macro = None
        self.iou = None

        os.makedirs(output_path, exist_ok=True)
        self.output_path = output_path

    def add(self, pred_frames_label):
    
        if not isinstance(pred_frames_label, list):
            pred_frames_label = list(pred_frames_label)
        
        if self.gt_label is not None and len(pred_frames_label) >= len(self.gt_label):
            self.pred_frames_label = (np.array(pred_frames_label)[:len(self.gt_label)]).astype(int)
        elif self.gt_label is not None:
            self.pred_frames_label = np.concatenate([pred_frames_label + [pred_frames_label[-1]] * (len(self.gt_label)-len(pred_frames_label))], axis=0).astype(int)

        else:
            self.pred_frames_label = np.array(pred_frames_label).astype(int)
            
        # Set the prediction to span successive integers (required for the Hungarian matching)
        mapping_original_ordinal_pred = {original: ordinal for ordinal, original  in enumerate(np.unique(self.pred_frames_label))}
        self.pred_frames_label = np.array([mapping_original_ordinal_pred[l] for l in self.pred_frames_label.astype(int)])

        self.prediction_is_available = True 

        return   

    def compute_metrics(self):

        if self.gt_label is None:
            print("No ground truth added.")
            return 

        if self.pred_frames_label is None:
            print("No prediction added.")
            return 


        # Find best assignment through Hungarian Method
        cost_matrix = estimate_cost_matrix(self.gt_label, self.pred_frames_label)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # decode the predicted labels
        self.pred_frames_label = col_ind[self.pred_frames_label]



        #-------------------- Compute Frame-wise metrics ------------------ #
        
        # Calculate the metrics (External libraries)
        self.accuracy = metrics.accuracy_score(self.gt_label, self.pred_frames_label)

        # F1-Score
        self.f1_macro = metrics.f1_score(self.gt_label, self.pred_frames_label, average='macro')  

        # Jaccard index or iou
        self.iou = np.sum(metrics.jaccard_score(self.gt_label, self.pred_frames_label, average=None)) / len(np.unique(self.gt_label))

        # Compute edit distance 
        self.edit = levenstein(self.pred_frames_label, self.gt_label)

        #-------------------- Compute Segment-wise metrics ------------------ #
        self.f1_10, self.f1_25, self.f1_50 = self.compute_segmental_f1(self.gt_label, self.pred_frames_label, overlaps=[0.1, 0.25, 0.5])

        return

    

    def compute_segmental_f1(self, frame_gt, frame_prediction, overlaps):

        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        # Add tp, fp and fn for f1 computation
        for s in range(len(overlaps)):
            tp1, fp1, fn1 = f_score(frame_prediction, frame_gt, overlaps[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
        f1s = np.array([0, 0 ,0], dtype=float)
        for s in range(len(overlaps)):
            precision = tp[s] / float(tp[s] + fp[s])
            recall = tp[s] / float(tp[s] + fn[s])

            f1 = 2.0 * (precision * recall) / (precision + recall)

            f1 = np.nan_to_num(f1) * 100
            f1s[s] = f1

        return f1s[0], f1s[1], f1s[2]

    def plot(self, names=None, cpts=[]):
        
        if self.gt_label is not None and self.pred_frames_label is not None:
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 5))
            ax1.imshow(np.repeat(self.gt_label[np.newaxis, :], 100, axis=0), cmap='tab20');ax1.set_title('Ground Truth Segmentation')
            ax1.scatter([0], [0], color='tab:blue', label='Background');ax1.legend()
            ax2.imshow(np.repeat(self.pred_frames_label[np.newaxis, :], 100, axis=0), cmap='tab20');ax2.set_title('Predicted Action segmentation')
            ax2.scatter([0], [0], color='tab:blue', label='Background');ax2.legend()
            plt.savefig(os.path.join(self.output_path, 'output.png'), dpi=80)
            plt.show() 
            
        elif self.pred_frames_label is not None:
            
            plt.figure(figsize=(30, 4))
            plt.imshow(np.repeat(self.pred_frames_label[np.newaxis, :], 200, axis=0), cmap='tab20');plt.title('Our segmentation')
            plt.scatter([0], [0], color='tab:blue', label='Background');plt.legend()
            plt.savefig(os.path.join(self.output_path, 'output.png'), dpi=80)
            plt.show() 
        
        return 

    def populate_ground_truth(self, df, n_frames=None):

        self.df = df
        gt_label = np.zeros(n_frames)

        for i, row in self.df.sort_values(by='start').iterrows():

            gt_label[int(row['start']):int(row['stop'])] = row['label']

        self.gt_label = np.array(gt_label).astype(int)
        self.mapping_original_ordinal_gt = {original: ordinal for ordinal, original  in enumerate(np.unique(self.gt_label))}
        self.gt_label = np.array([self.mapping_original_ordinal_gt[l] for l in self.gt_label])
        self.gt_available=True
        return 

def estimate_cost_matrix(gt_label, cluster_labels):
    # Make sure the lengths of the inputs match:
    if len(gt_label) != len(cluster_labels):
        print('The dimensions of the gt_labls and the pred_labels do not match')
        return -1
    L_gt = np.unique(gt_label)
    L_pred = np.unique(cluster_labels)
    nClass_pred = len(L_pred)
    dim_1 = max(nClass_pred, np.max(L_gt) + 1)
    profit_mat = np.zeros((nClass_pred, dim_1))
    
    for i, frame_pred in enumerate(L_pred):
        idx = np.argwhere(cluster_labels == frame_pred)
        gt_selected = np.array(gt_label)[idx]
        for j, frame_gt in enumerate(L_gt):
            profit_mat[i][j] = np.count_nonzero(gt_selected == frame_gt)
            
    return -profit_mat




def f_score(recognized, ground_truth, overlap, bg_class=[0]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
 
    tp = 0
    fp = 0
 
    hits = np.zeros(len(y_label))
 
    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()
 
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)

def levenstein(frame_prediction, frame_gt, norm=True):

    p, _, _ = get_labels_start_end_time(frame_prediction, bg_class=[0])
    y, _, _ = get_labels_start_end_time(frame_gt, bg_class=[0])

    n_clusters = len(np.unique(y))
    
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                            D[i, j-1] + 1,
                            D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score    

def get_labels_start_end_time(frame_wise_labels, bg_class=[0]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends