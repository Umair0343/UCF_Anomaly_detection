import os
import numpy as np
import torch
from torch.utils.data import Dataset
from itertools import cycle

class UCFDataset(Dataset):
    def __init__(self, data_dir, test=False, annotation_file=None):
        self.data_dir = data_dir
        self.class_folders = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        self.test = test
        self.annotation_file = annotation_file

        self.file_paths = []
        for class_folder in self.class_folders:
            class_dir = os.path.join(data_dir, class_folder)
            file_paths = [
                os.path.join(class_dir, file_name)
                for file_name in os.listdir(class_dir)
                if file_name.endswith("_x264.npy") and not file_name.startswith(".")
            ]
            self.file_paths.extend(file_paths)

        if self.test and self.annotation_file:
            self.annotations = self.load_annotations(self.annotation_file)

    def load_annotations(self, annotation_file):
        annotations = {}
        with open(annotation_file, 'r') as f:
            for line in f:
                video_name, event, start_frame1, end_frame1, start_frame2, end_frame2 = line.strip().split()
                start_frame1 = int(start_frame1) if start_frame1 != "-1" else -1
                end_frame1 = int(end_frame1) if end_frame1 != "-1" else -1
                start_frame2 = int(start_frame2) if start_frame2 != "-1" else -1
                end_frame2 = int(end_frame2) if end_frame2 != "-1" else -1
                annotations[video_name] = (event, start_frame1, end_frame1, start_frame2, end_frame2)
        return annotations

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        video_crops = np.load(file_path)
        labels = np.zeros(len(video_crops))

        if self.test:
            video_name = os.path.splitext(os.path.basename(file_path))[0] + ".mp4"

            if video_name in self.annotations:
                event, start_frame1, end_frame1, start_frame2, end_frame2 = self.annotations[video_name]
                init_labels = np.zeros((len(video_crops) * 16))
                if start_frame1 != -1:
                    init_labels[start_frame1-1: end_frame1] = 1
                if start_frame2 != -1:
                    init_labels[start_frame2-1: end_frame2] = 1

                segment_size = 16
                for i in range(len(video_crops)):
                    segment = init_labels[i * segment_size: (i + 1) * segment_size]
                    if np.sum(segment) >= 7:
                        labels[i] = 1

            seg = torch.tensor(video_crops)
            lbl = torch.tensor(labels)

        else:
            video_name = os.path.splitext(os.path.basename(file_path))[0]

            if video_name.startswith("Normal"):
                labels = np.zeros(len(video_crops))
            else:
                labels = np.ones(len(video_crops))

            batch_size = 64
            num_segments = len(video_crops)
            seg = []
            lbl = []

            for start_idx in range(0, num_segments, batch_size):
                end_idx = start_idx + batch_size
                selected_segment = video_crops[start_idx:end_idx]
                selected_label = labels[start_idx:end_idx]

                # Pad to batch size if needed
                if len(selected_segment) < batch_size:
                    selected_segment = [item for item, _ in zip(cycle(selected_segment), range(batch_size))]
                    selected_label = [item for item, _ in zip(cycle(selected_label), range(batch_size))]

                selected_segments = torch.from_numpy(np.array(selected_segment))
                selected_labels = torch.from_numpy(np.array(selected_label))

                seg.append(selected_segments)
                lbl.append(selected_labels)

        return seg, lbl