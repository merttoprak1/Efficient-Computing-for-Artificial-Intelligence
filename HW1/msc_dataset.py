import os
import torch
import torchaudio
from torch.utils.data import Dataset

class MSCDataset(Dataset):
    # A custom dataset loader for the Mini Speech Commands data.
    # This handles finding the files and formatting them so PyTorch can use them.
    def __init__(self, root: str, classes: list):
        # We need the folder path and the list of class names to set everything up.
        self.root = root
        self.classes = classes
        
        # Get a list of every .wav file in the folder so we can index them later.
        self.files = [f for f in os.listdir(root) if f.endswith('.wav')]
        
        # Neural nets handle numbers better than strings, so we make a quick dictionary
        # to map labels like 'down' or 'stop' to integers
        self.label_to_int = {label: i for i, label in enumerate(self.classes)}

    def __len__(self):
        # How many total items we have to calculate epochs
        return len(self.files)

    def __getitem__(self, idx):
        # This is the main workhorse. It fetches the specific item at index 'idx'
        
        # Figure out the full path to the file we need.
        filepath = os.path.join(self.root, self.files[idx])

        # Load the audio. torchaudio is great because it returns the waveform
        # directly as a tensor, saving us a conversion step.
        waveform, sample_rate = torchaudio.load(filepath)

        # The label is hidden inside the filename 
        # Need the part before the underscore
        label_str = self.files[idx].split('_')[0]
        
        # Convert that string label into its integer ID using the map we made earlier.
        label_int = self.label_to_int[label_str]

        # Get everything into a dictionary.
        # This structure makes it easy to unpack inside the training loop.
        sample = {
            "x": waveform,
            "sampling_rate": sample_rate,
            "label": label_int
        }
        
        return sample