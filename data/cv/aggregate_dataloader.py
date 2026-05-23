from torch.utils.data import Dataset, ConcatDataset
from data.cv.ego4d_dataloader import Ego4dDataset
from data.cv.finevideo_dataloader import FineVideoDataset
from data.cv.imagenet_dataloader import ImageNetDataset, ImageNetVideoDataset
from data.cv.kinetics_dataloader import *
from data.cv.something_dataloader import *

class AggregateDataset(Dataset):
    def __init__(self, hparams, split, transform):
        self.name_to_class = {
            'smth': SomethingDataset,
            'something': SomethingDataset,
            'ssv2': SomethingDataset,
            'smth': SomethingDataset,
            
            'ego4d': Ego4dDataset,
            
            'kinetics400': Kinetics400Dataset,
            'k400': Kinetics400Dataset,

            'imagenet-video': ImageNetVideoDataset,

            'finevideo': FineVideoDataset,
        }

        self.datasets = []
        names = [name.strip().lower() for name in hparams.aggr_datasets_list.split(',')]
        directories = [dir.strip() for dir in hparams.aggr_dataset_dirs.split(',')]
        
        # if theres three directories, but only two names, we will only use the two names and ignore the third directory
        assert len(names) <= len(directories), "More dataset names found than the provided dataset directories."

        for dataset_name, dir in zip(names, directories):
            if dataset_name in self.name_to_class:
                dataset = self.name_to_class[dataset_name](hparams, dataset_dir=dir, split=split, transform=transform)
                self.datasets.append(dataset)
                print(f"Loaded {dataset_name} dataset with {len(dataset)} samples for {split} split.")

            else:
                raise ValueError(f"Dataset {dataset_name} not supported for aggregation. Available datasets: {list(self.name_to_class.keys())}")
            
        self.aggregate_ds = ConcatDataset(self.datasets)
            
    def __len__(self):
        return len(self.aggregate_ds)

    def __getitem__(self, idx):
        return self.aggregate_ds[idx]
        
    
