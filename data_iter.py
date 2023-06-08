import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

# ===========================
# Build a dataset, inherited the methods of Dataset, that returns tensors for Filler
class FillerDataset(Dataset):
    
    def __init__(
        self, 
        data, 
        tokenizer
    ): 
        # The data is from the preprocessed file, which contains (scaffold, decorations, smiles)
        self.data = data
        self.scaffold_smi = self.data['scaffold']
        self.decorations_smi = self.data['decorations']
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        scaffold = self.scaffold_smi[idx]
        decorations = self.decorations_smi[idx].strip()
        self.scaffold = self.tokenizer.scaffold_encode(scaffold)
        self.decorations = self.tokenizer.decoration_encode(decorations)

        return self.scaffold, self.decorations

# ===========================
# Definite the data loader for Filler model
class FillerDataLoader(LightningDataModule):

    def __init__(
        self, 
        tokenizer,
        preprocessed_file, 
        train_size=10000, 
        val_size=200,
        batch_size=64
    ):
        
        super().__init__()
        self.tokenizer = tokenizer
        self.preprocessed_file = preprocessed_file
        self.train_size = train_size
        self.val_size = val_size
        self.batch_size = batch_size
        
    # Pad  and generate segment_ids and offsets_ids for the output of FillerDataset.getitem()
    def segment_offsets_and_pad(self, batch):
        """
        batch: a list of vectorized smiles [self.scaffold, self.decorations]
        return: 
            -scaffold: a dictionary and the names are seq_ids, segment_ids, offsets_ids, mask_segment_ids
            -decorations: a dictionary and the names are seq_ids, segment_ids, offsets_ids
        """
        scaffold_lists = [sca for sca, _ in batch]
        # Get the seq_ids, segment_ids, offsets_ids, and mask_segment_ids for the scaffold
        self.scaffold_outputs = self.get_scaffold_segment_and_offsets_ids(scaffold_lists) 
        decorations = [dec for _, dec in batch] 
        # Get the seq_ids, segment_ids, and offsets_ids for the decorations
        self.decoration_outputs = self.get_decoration_segment_and_offsets_ids(
            self.scaffold_outputs['mask_segment_ids'], 
            decorations
        )

        return self.scaffold_outputs, self.decoration_outputs
    
    # Generate the seq_ids, segment_ids, offsets_ids and mask_segment_ids for a batch size of the scaffolds
    def get_scaffold_segment_and_offsets_ids(self, scaffold_lists):
        """
        scaffold_lists: contain the start and end tokens
        return: outputs
        """
        # Cover the scaffold_lists into tensors
        tensors = [torch.tensor(sca) for sca in scaffold_lists]
        # Pad the different lengths of tensors to the maximum length (each row is a sequence) [batch_size, seq_len]
        tensors = torch.nn.utils.rnn.pad_sequence(
            tensors, 
            batch_first=True, 
            padding_value=self.tokenizer.char_to_int[self.tokenizer.pad]
        ) 
        # Intialize the segment_ids, offsets_ids and mask_segment_ids
        segment_ids =  torch.zeros(tensors.size(), dtype=torch.int64)
        offsets_ids = torch.zeros(tensors.size(), dtype=torch.int64)
        mask_segment_ids = []
        
        for i in range(tensors.size(0)): # batch_size
            segment_flag = 0
            offset_flag = 0
            mask_seg_id = []
            
            for j in range(tensors.size(1)): # maxlength
                if tensors[i][j] == self.tokenizer.char_to_int['*']:
                    segment_flag += 1
                    offset_flag = 0
                    segment_ids[i][j] = segment_flag
                    mask_seg_id.append(segment_flag)
                    segment_flag += 1
                else:
                    segment_ids[i][j] = segment_flag
                    offsets_ids[i][j] += offset_flag
                    offset_flag += 1
            mask_segment_ids.append(mask_seg_id)
        
        outputs = {
            'seq_ids': tensors,
            'segment_ids': segment_ids,
            'offsets_ids': offsets_ids,
            'mask_segment_ids': torch.tensor(mask_segment_ids)
        }
        
        return outputs
 
    # Generate the seq_ids, segment_ids, and offsets_ids for a batch size of the decorations
    def get_decoration_segment_and_offsets_ids(self, mask_ids, decorations):
        """
        mask_ids: scaffold outputs['mask_segment_ids'] for segment_ids
        decorations: list type of decorations
        return: outputs
        """
        # Perform the first column of the decorations, then the second for a batch size of decorations
        decoration_col = [torch.tensor(row) for row in decorations]
        tensors = torch.nn.utils.rnn.pad_sequence(
            decoration_col, 
            batch_first=True, 
            padding_value=self.tokenizer.char_to_int[self.tokenizer.pad]
        ) # [batch_size, maxlength]
        # Intialize the segment_ids and offsets_ids
        offsets_ids = torch.arange(0, tensors.size(1)).repeat(tensors.size(0), 1)
        segment_ids = torch.tensor(np.array(mask_ids).reshape(tensors.size(0), 1)).repeat(1, tensors.size(1))
        
        outputs = {
            'seq_ids': tensors,
            'segment_ids': segment_ids,
            'offsets_ids': offsets_ids
        }
            
        return outputs
    
    def setup(self):
        self.data = pd.read_csv(self.preprocessed_file, nrows = self.val_size + self.train_size, sep=';', names = ['scaffold', 'decorations', 'smiles'])
        self.tokenizer.build_vocab()
        idxs = list(range(len(self.data['scaffold'])))
        np.random.shuffle(idxs)
        val_idxs, train_idxs = idxs[:self.val_size], idxs[self.val_size:self.val_size + self.train_size]
        
        # Split train and validation datasets
        self.train_data = self.data.loc[train_idxs]
        self.train_data.reset_index(drop=True, inplace=True)
        self.val_data = self.data.loc[val_idxs]
        self.val_data.reset_index(drop=True, inplace=True)
        
    def train_dataloader(self):
        dataset = FillerDataset(self.train_data, self.tokenizer)
        
        return DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size, 
            pin_memory=True,  # pin_memory=True: speed the dataloading, num_workers: multithreading for dataloading
            collate_fn=self.segment_offsets_and_pad, 
            shuffle=True,
            num_workers=0 
        ) 
    
    def val_dataloader(self):
        dataset = FillerDataset(self.val_data, self.tokenizer)
        
        return DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size, 
            pin_memory=True, 
            collate_fn=self.segment_offsets_and_pad, 
            shuffle=True, 
            num_workers=0
        ) 

# ===========================
# Build a dataset, inherited the methods of Dataset, that returns tensors for Discriminator
class DisDataset(Dataset):
    
    def __init__(self, pairs, tokenizer):
        """
        pairs: contain smiles and labels
        tokenizer: a class object
        """
        self.data, self.labels = pairs['smiles'], pairs['labels']
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smi = self.data[idx]
        tensor = self.tokenizer.scaffold_encode(smi)[1: -1] # Remove the sequence start and end tokens
        label = self.labels[idx]
        
        return tensor, label
        
# ===========================
# Define the data loader for Discriminator
class DisDataLoader(LightningDataModule):
    
    def __init__(
        self, 
        postive_file, 
        negative_file, 
        tokenizer, 
        batch_size=64
    ):
        """
        postive_file: the third column ('smiles') of the original preprocessed data
        negative_file: the generated smiles data file
        tokenizer: a class object
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.postive_file = postive_file
        self.negative_file = negative_file
    
    def custom_collate_and_pad(self, batch):
        smiles, labels = zip(*batch)
        tensor_smiles = [torch.LongTensor(smi) for smi in smiles]
        tensors = torch.nn.utils.rnn.pad_sequence(
            tensor_smiles, 
            batch_first=True, 
            padding_value=self.tokenizer.char_to_int[self.tokenizer.pad]
        ) # [batch_size, maxlength]
        labels = torch.LongTensor(labels)
        
        return tensors, labels
        
    def setup(self):
        # Load postive and negative data
        self.postive_data = pd.read_csv(self.postive_file, sep=';', names=['scaffold', 'decorations', 'smiles'])
        self.negative_data = pd.read_csv(self.negative_file, sep=';', names=['scaffold', 'decorations', 'smiles'])
        # Keep the canonical order for the negative dataset
        self.data = pd.concat([self.postive_data['decorations'], self.negative_data['decorations']])
        self.labels = pd.DataFrame([1 for _ in range(len(self.postive_data))] + [0 for _ in range(len(self.negative_data))], columns=['labels'])
        self.pairs = list(zip(self.data, self.labels['labels']))
        # Shuffle the input data for the discriminator
        np.random.shuffle(self.pairs) 
        self.pairs = pd.DataFrame(self.pairs, columns=['smiles', 'labels'])
        self.train_data = self.pairs[:int(len(self.pairs)*0.9)]
        self.val_data = self.pairs[len(self.train_data):]
        self.train_data.reset_index(drop=True, inplace=True)
        self.val_data.reset_index(drop=True, inplace=True)
        
    def train_dataloader(self):
        dataset = DisDataset(self.train_data, self.tokenizer)
        
        return DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=self.custom_collate_and_pad,
            shuffle=True,
            num_workers=0
        )
    
    def val_dataloader(self):
        dataset = DisDataset(self.val_data, self.tokenizer)
        
        return DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size,
            pin_memory=True,
            collate_fn=self.custom_collate_and_pad,
            shuffle=True,
            num_workers=0
        )
    






    
