import os
import numpy as np
import pandas as pd
import os
import warnings
import time
warnings.filterwarnings("ignore")

# Import our own algorithms
from src.dataset import VideoFrameDataset
from src.pipeline import Pipeline
from src.metrics import Annotation
from src.const import DEFAULT_CONFIG



# if you need to access a file next to the source code, use the variable ROOT
# for example:
#    torch.load(os.path.join(ROOT, 'weights.pth'))
ROOT = os.path.dirname(os.path.realpath(__file__))

def main(video_path, ground_truth_path, num_clusters):
    start_time = time.time()

    # Create the video dataset
    dataset = VideoFrameDataset(filename=video_path, config=DEFAULT_CONFIG)
    
    
    # Create the ground truth and metrics computation object
    annotation = Annotation()
    
    if ground_truth_path is not None:
        df = pd.read_csv(ground_truth_path)
        annotation.populate_ground_truth(df, dataset.n_frames)

    # Create the pipeline object
    pipeline = Pipeline(dataset=dataset, annotation=annotation, config=DEFAULT_CONFIG, verbosity=0)
    
    # Compute frame embedding
    pipeline.compute_embedding()
        
        
    # Do the background extraction
    pipeline.outlier_methods = ['custom', 'silhouette_score']
    pipeline.remove_outliers_method()

    # Calibration of the lambda parameter 
    regularization_lambda = int(10*3000/dataset.n_frames)

    # Detect ruptures
    pipeline.detect_ruptures(penalty=regularization_lambda, ruptures_on = 'embedding_ts', remove_outliers=True, verbose=False)

    
    if ground_truth_path is not None:
        # Take as number of cluster the number of different actions in the ground truth
        num_clusters = len(np.unique(annotation.gt_label)) - 1
        
    else:
        num_clusters = num_actions
        


    # Cluster non-background segments
    pipeline.cluster_frames(n_clusters=num_clusters, method='kmeans')

    
    if ground_truth_path is not None:
        pipeline.annotation.compute_metrics()

        # Create outputs
        pipeline.create_outputs()

        print("Frame-wise accuracy: {:.2f}".format(pipeline.annotation.accuracy))
        print("Frame-wise F1-score (macro averaging): {:.2f}".format(pipeline.annotation.f1_macro))
        print("Jaccard index: {:.2f}".format(pipeline.annotation.iou))
        print("Edit score (Levenstein metric): {:.2f}".format(pipeline.annotation.edit))
        print("Segment-wise F1@.10: {:.2f}".format(pipeline.annotation.f1_10))
        print("Segment-wise F1@.25: {:.2f}".format(pipeline.annotation.f1_25))
        print("Segment-wise F1@.50: {:.2f}".format(pipeline.annotation.f1_50))
        
    print("--- Runtime: %s seconds ---" % (time.time() - start_time))
    print("Done! You can find the output png of the segmentation in ./outputs/output.png, the performance in ./outputs/performances.csv, and segmentation in ./outputs/prediction.csv")
    return 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=False, default='./data/sample_video_ground_truth.csv')
    parser.add_argument("--ground_truth_path", type=str, required=False, default=None)
    parser.add_argument("--num_actions", type=int, required=False, default=8)

    args = parser.parse_args()
    main(args.video_path, args.ground_truth_path, args.num_actions)
