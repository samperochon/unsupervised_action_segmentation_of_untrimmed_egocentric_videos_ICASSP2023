import os
import sys
import numpy as np
import imageio

  
class VideoFrameDataset(object):
    """Dataset used to get frames form a video."""

    def __init__(self, filename, config):
        """
        Args:
            filename (string)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        def find_n_frames():
            for i in range(self.video.count_frames(), 0, -1):
                try:
                    self.video.get_data(i)
                    break
                except:
                    pass
            return i-1
        
        self.filename = filename
        self.video_name = os.path.basename(filename)[:-4]

        self.video = imageio.get_reader(filename,  'ffmpeg')
        self.n_frames = find_n_frames()
        self.frame_size = self.video.get_meta_data()['source_size']
        self.fps = self.video.get_meta_data()['fps']
        self.duration = self.video.get_meta_data()['duration']
        self.config = config
        self.verbosity = config['verbosity']

        print("Init. video {} with {} frames".format(self.video_name, self.n_frames)) if self.verbosity > 2 else None
        
        return
                

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx):
        frame = self.video.get_data(idx)
        return frame

