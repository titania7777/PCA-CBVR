import os
import sys
import cv2
import ffmpeg
import inspect
from glob import glob
from joblib import Parallel, delayed

from .Util import PathManager, PrintManager

class FrameExtractor(object):
    def __init__(self, video_path:str, save_path:str, frame_size:int=-1, num_workers:int=-1):

        _, _, _, self.args_dict = inspect.getargvalues(inspect.currentframe())
        self.video_path = video_path
        self.save_path = save_path
        self.frame_size = frame_size # -1 -> original size
        self.num_workers = num_workers # -1 -> using all cores
        self.start_point_of_path = len(os.path.join(self.video_path, "").split("/")) - 1
        self.video_path_list = glob(os.path.join(self.video_path, "**/*.*"), recursive=True)
        self.num_videos = len(self.video_path_list)

    def run(self):
        # path check
        PathManager(self.video_path).exist(raise_error=True)
        save_path_manager = PathManager(self.save_path)
        if save_path_manager.exist(raise_error=False):
            if save_path_manager.remove(enforce=False):
                save_path_manager.create()
            else:
                return
        else:
            save_path_manager.create()
            
        # display arguments
        PrintManager(self.save_path).args(args_dict=self.args_dict)

        # run
        Parallel(n_jobs=self.num_workers, backend="threading")(
            delayed(self._rgb)(
                [i, self.num_videos], video_path, self.start_point_of_path, self.save_path, self.frame_size
            ) for i, video_path in enumerate(self.video_path_list)
        );print()

    def _rgb(self, index:list, video_path:str, start_point_of_path:int, save_path:str, frame_size:int):
        # get video name and save path
        video_name, save_path = self._get_video_name_save_path(start_point_of_path, video_path, save_path)

        # get frame information
        length, original_width, original_height = self._get_frame_info(video_path)

        if frame_size == -1:
            resized_width, resized_height = original_width, original_height
        else:
            resized_width, resized_height = self._get_resized_size(original_width, original_height, frame_size)

        # message
        sys.stdout.write(f"\r{index[0]+1}/{index[1]} ({original_width}x{original_height}) -> ({resized_width}x{resized_height}) length: {length:<{5}} name: {video_name}")

        # run ffmpeg
        (
            ffmpeg.input(video_path)
            .filter("scale", resized_width, resized_height)
            .output(os.path.join(save_path, "%d.jpeg"))
            .global_args("-loglevel", "error", "-threads", "1", "-nostdin")
            .run()
        )
    
    # TODO
    def _flow(self): pass
    
    def _get_video_name_save_path(self, start_point_of_path:int, video_path:str, save_path:str):
        split_video_path = video_path.split("/")
        video_name = split_video_path[-1].split(".")[0]
        save_path = os.path.join(save_path, *split_video_path[start_point_of_path:-1], video_name)
        os.makedirs(save_path)
        return video_name, save_path

    def _get_frame_info(self, video_path:str):
        # read video information
        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return length, width, height
    
    # resizing the frame size based on small values
    # ignore resizing if the frame size is small than the input size
    def _get_resized_size(self, width:int, height:int, frame_size:int):
        if width > height:
            aspect_ratio = width / height
            if height >= frame_size:
                height = frame_size
            width = int(aspect_ratio*height)
        else:
            aspect_ratio = height / width
            if width >= frame_size:
                width = frame_size
            height = int(aspect_ratio*width)
        return width, height