import cv2
import os, shutil
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import subprocess
import json
import pandas
from data.utils.data_preprocessor import remove_padding, crop_to_square, get_all_frames, handle_corrupt_file

class SomethingDataset(Dataset):
    def __init__(self, hparams, split, transform, dataset_dir = None, labels_dir = 'labels/', ignore_filepath=""):
        """
        Args:
            dataset_dir (str): Directory containing class folders.
            transform (callable, optional): Optional transform to apply to frames.
            num_frames (int): Number of frames to return from each video.
        """
        self.current_user = os.getenv("USER")
        self.hparams = hparams
        self.dataset_dir = dataset_dir if dataset_dir is not None else hparams.dataset_dir
        self.labels_dir = self.dataset_dir + '/' + labels_dir
        self.transform = transform
        self.num_frames = self.hparams.context_length
        self.time_between_frames = self.hparams.time_between_frames
        self.sampling_rate = self.hparams.sampling_rate
        self.split = split
        
        # This file is the output of ebwm/utils/find_corrupt_files.py
        # Filepath should be .../ebwm/utils/corrupt_files/<dataset>.txt
        self.potential_ignore_filepath = '/'.join([os.getcwd(), "data/cv/corrupt_files/ssv2.txt"])
        if ignore_filepath:
            self.ignore_filepath = self.labels_dir + ignore_filepath
        elif os.path.exists(self.potential_ignore_filepath):
            self.ignore_filepath = self.potential_ignore_filepath
        else:
            self.ignore_filepath = ignore_filepath

        if self.hparams.preencode_dataset or self.hparams.use_preencoded_dataset:
            self.hdf5_file = None   # lazy load only when retrieving frames
        
        # FFPROBE SETUP
        # First, check if ffprobe is in path
        if shutil.which("ffprobe") is not None:
            self.ffprobe = shutil.which("ffprobe")
        # If not, check if environment variable was set
        elif os.getenv("FFPROBE_PATH") is not None:
            self.ffprobe = os.getenv("FFPROBE_PATH")
        # Finally, check if hparam is set
        elif hparams.ffprobe_path is not None:
            self.ffprobe = hparams.ffprobe_path
        else:
            # Default to scratch user directory
            self.ffprobe = f'/scratch/{self.current_user}/ffprobe'
        assert os.path.exists(self.ffprobe)
        
        self.class_names = []
        self.file_list = []
        self.labels = []
        self.frame_lookup = None
        
        if self.split not in ('train','test', 'val'):
            raise(Exception("Split must be train, test, or val"))
        
        annotations = None
        if self.split == 'val':
            annotations = pandas.read_json(f"{self.labels_dir}/validation.json")
        else:
            annotations = pandas.read_json(f"{self.labels_dir}/{self.split}.json")        

        label_dict = pandas.read_json(f"{self.labels_dir}/labels.json", typ='Series')

        if self.split == 'test':
            labels = pandas.read_csv((f"{self.labels_dir}/test-answers.csv"), header=None, sep=';')
            labels.columns = ['id', 'label']
            labels.set_index('id')
            self.labels = list(label_dict[labels.label])
        else:                  
            self.labels = list(label_dict[annotations.apply(lambda r: r.template.replace("[","").replace("]",""), axis=1)])
        self.class_names = list(label_dict.values)
        
        self.file_list = list(annotations.apply(lambda r: 
        f"{self.dataset_dir}/20bn-something-something-v2/{r.id}.webm", axis=1))

        self.video_ids = list(annotations.apply(lambda r: f"{r.id}.webm", axis=1))

        # print("Filtering out non-existent video paths for SSV2")
        # self.file_list = list(filter(os.path.exists, self.file_list))

        # TODO: report bug, this removes the dataset dir
        # if self.ignore_filepath:
        #     with open(self.ignore_filepath) as f:
        #         self.ignore_files = f.read()
        #     new_file_list = []
        #     for fp in self.file_list:
        #         fp_components = os.path.normpath(fp).split('/')
        #         new_fp = '/'.join(fp_components[-2:]) 
        #         if new_fp not in self.ignore_files:
        #             new_file_list.append(new_fp)
        #     self.file_list = new_file_list

        
        
    def get_length(self, filename):
        command = self.ffprobe + ' -v quiet -print_format json -show_format "{}"'.format(filename)
        data = json.loads(subprocess.check_output(command, shell=True))
        return float(data['format']['duration'])


    def get_num_frames(self, filepath):
        cap = cv2.VideoCapture(filepath)
        estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        count=0
        for i in range(estimate):
            ret, img = cap.read()
            if not ret: break
            count += 1
            
        return count
    

    def get_num_frames_static(self, filepath):
        if self.frame_lookup == None:
            self.frame_lookup = json.load(open(self.labels_dir+"lengths.json"))
        
        return self.frame_lookup[filepath]
        

    def __len__(self):
        if self.hparams.debug_mode:
            return 1000
        else:
            return len(self.file_list)
        

    def _init_hdf5(self):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hparams.hdf5_file_path, "r")


    def __del__(self):
        if hasattr(self, 'hdf5_file') and self.hdf5_file:
            self.hdf5_file.close()


    def __getitem__(self, idx):
        video_path = self.file_list[idx]
        video_id = self.video_ids[idx]

        label = self.labels[idx]

        try:         
            video_length = self.get_length(video_path)
            total_frames = self.get_num_frames(video_path)-1


            # -- randomly pick out required frames
            frame_idx_list = self.get_frame_indices(video_length, total_frames)
            crops = self.capture_video_frames(video_path, self.num_frames, frame_idx_list)

            # prev_crops = crops[:-1]
            # next_crops = crops[1:]

            # prev_frames = [torch.stack(same_crops) for same_crops in zip(*prev_crops)]
            # next_frames = [torch.stack(same_crops) for same_crops in zip(*next_crops)]

            # return (prev_frames, next_frames), "dummy"

            frames = [torch.stack(same_crops) for same_crops in zip(*crops)]
            return frames, "dummy"
        
            
        except Exception as exception:
            log_path = self.ignore_filepath if self.ignore_filepath else self.potential_ignore_filepath
            handle_corrupt_file(exception, video_path, log_path)
            
            # recursion: try next sample
            return self[random.randint(0, len(self)-1)]


    def capture_video_frames(self, video_path, num_frames, frame_idx_list):
        cap = cv2.VideoCapture(video_path)
        frames = []
        for i in range(num_frames):           
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_list[i].item())
            ret, frame = cap.read()
            if not ret: 
                print(f"frame is blank at {frame_idx_list[i].item()}")

            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_width, original_height = original_frame.shape[:2]
            pil_image = Image.fromarray(original_frame)
                
            if self.hparams.preprocess_data:
                frame, cropped = remove_padding(original_frame)  # Assuming this function is defined
                if frame.shape[0] < original_height / 2 or frame.shape[1] < original_width / 2:
                        # Call remove_padding again with different threshold values
                    frame, _ = remove_padding(frame, threshold_value=10, threshold_ratio=0.9)
                    if frame.shape[0] < original_height / 2 or frame.shape[1] < original_width / 2:
                        frame, _ = remove_padding(frame, threshold_value=10, threshold_ratio=0.95)
                            #code below is for debugging
                            # if frame.shape[0] < original_height / 2 or frame.shape[1] < original_width / 2:
                            #     save_path = f"./logs/debug/images/special/{video_path}.png"  # Specify the path where you want to save the image
                            #     cv2.imwrite(save_path, cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR))

                if self.hparams.crop_all_samples or cropped:
                    frame = crop_to_square(frame)  # Assuming this function is defined, we crop all images generally since dont want non uniform distortion

                #code below is for debugging########################################################################
                    # cropped_path = f"./logs/debug/images/{idx}_cropped.png"  # Specify the path where you want to save the image
                    # cv2.imwrite(cropped_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    # uncropped_path = f"./logs/debug/images/{idx}_uncropped.png"  # Specify the path where you want to save the image
                    # cv2.imwrite(uncropped_path, cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR))
                ##########################################################################################
 
            if self.transform:
                transformed_frame = self.transform(pil_image)
                frames.append(transformed_frame)
            else:
                raise ValueError("Please specify transform")

        cap.release()

        return frames


    def get_frame_indices(self, video_length, total_frames):
        if total_frames < self.num_frames:
            frame_idx_list = torch.linspace(0, total_frames, steps=self.num_frames).long() # just duplicate frames
        elif self.hparams.use_raw_framerate:
            if self.hparams.no_randomness_dataloader:
                start_frame = 0
            else: # normal condition
                start_frame = random.random() * (total_frames-self.num_frames)
            frame_idx_list = torch.arange(start_frame, start_frame+self.num_frames).long()
        elif self.hparams.sampling_rate != 0:
            required_frames = self.hparams.sampling_rate * (self.num_frames-1)
            if required_frames > total_frames:
                    # print("required_frames was > total_frames", required_frames, total_frames)
                frame_idx_list = torch.linspace(0, total_frames, steps=self.num_frames).long() # Default case when you can't put the desired space between frames
            else:
                start_frame = int(random.random() * (total_frames-required_frames))
                frame_idx_list = torch.linspace(start_frame, start_frame+required_frames, steps=self.num_frames).long()
        else: # uses time_between_frames hparam
            required_length = self.time_between_frames * (self.num_frames-1)
            if required_length > video_length:
                frame_idx_list = torch.linspace(0, total_frames, steps=self.num_frames).long() # Default case when you can't put the desired space between frames
            else:
                start_time = random.random() * (video_length-required_length)
                end_time = start_time + required_length

                frame_idx_list = (total_frames/video_length
                                    *torch.linspace(start_time, end_time, steps=self.num_frames)).long()
                                
        return frame_idx_list
