import math
import torch
import numpy as np
import pandas as pd
from utils import *
from rdkit import Chem
import pytorch_lightning
from pytorch_lightning import LightningModule

# ===========================
# Define the position embedding manually
class PositionEmbedding(torch.nn.Module):
    
    def __init__(
        self, 
        d_model, 
        segment_ids,
        offsets_ids,
        base=256,
        dropout=0.1,
    ):
        """
        d_model: the embedded dimension
        segment_ids: [batch_size, seq_len]
        offsets_ids: [batch_size, seq_len]
        """
        super(PositionEmbedding, self).__init__()
        self.dropout = torch.nn.Dropout(p = dropout)
        self.d_model = d_model

        position = segment_ids * base + offsets_ids # [batch_size, seq_len]
        position = position.unsqueeze(2) # [batch_size, seq_len, 1]
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(1e4) / self.d_model)).to(position.device) # 1/1e4^(2i/d_model)
        pe = torch.zeros(segment_ids.size(0), segment_ids.size(1), self.d_model) # [batch_size, seq_len, d_model]
        pe[:, :, 0::2] = torch.sin(position * div_term) # sin(pos/10000^(2i/d_model))
        pe[:, :, 1::2] = torch.cos(position * div_term) # cos(pos/10000^(2i/d_model))
        self.register_buffer('pe', pe) # Position embedding that does not need backpropagation
    
    def forward(self, embed):
        """
        embed: the embeded sequence with the tensor size of [batch_size, seq_len, d_model]
        return: the embedding with the tensor size of [batch_size, seq_len, d_model]
        """
        embed = embed + self.pe[:embed.size(0)].to(embed.device) #[batch_size, seq_len, d_model] 
        embed = self.dropout(embed)
        
        return embed
    
# ===========================
# Definition of the Filler mdoel who can infilling decorations for a given scaffold
class FillerModel(LightningModule):
    
    def __init__(
        self, 
        n_tokens,
        d_model=256,
        nhead=8,
        num_decoder_layers=4,
        dim_feedforward=200,
        dropout=0.1,
        activation='relu',
        max_lr=1e-4,
        epochs=200,
        train_size=10000,
        batch_size=64,
        optimizer='RMSprop'
    ):
        
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.max_lr = max_lr
        self.epochs = epochs
        self.train_size = train_size
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.setup_layers()
        self.steps_per_epoch =  math.ceil(self.train_size/self.batch_size)
    
    # Configure the optimizers for training and validation
    def configure_optimizers(self):    
        if self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.max_lr)
        else:
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.max_lr)
            
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.max_lr,
            total_steps=None, 
            epochs=self.epochs, 
            steps_per_epoch=self.steps_per_epoch,
            pct_start=4/self.epochs, 
            anneal_strategy='cos', 
            cycle_momentum=True, 
            base_momentum=0.85, 
            max_momentum=0.95,
            div_factor=1e3, 
            final_div_factor=1e3, 
            last_epoch=-1
        )
        
        scheduler = {'scheduler': scheduler, 'interval' : 'step'}
        
        return [optimizer], [scheduler]
        
    def setup_layers(self):
        self.embedding = torch.nn.Embedding(self.n_tokens, self.d_model)
        decoder_layer = torch.nn.TransformerDecoderLayer(
            self.d_model, 
            self.nhead, 
            self.dim_feedforward,
            self.dropout,
            self.activation,
            batch_first=True
        )
        decoder_norm = torch.nn.LayerNorm(self.d_model)
        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer, 
            self.num_decoder_layers, 
            decoder_norm
        )
        self.fully_connect_layer = torch.nn.Linear(self.d_model, self.n_tokens)
            
    # Define lower triangular square matrix with dim = sz
    def _generate_square_subsequent_mask(self, sz):
        """
        sz: an integer (target_len)
        return: a low triangular square matrix tensor[sz, sz]
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1) 
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        return mask
        
    def forward(
        self, 
        scaffold_input, 
        decoration_input
    ):
        """
        args:
            -scaffold_input: a dictionary
                (1) scaffold_input['seq_ids']: the scaffold sequence with the size of [batch_size, source_len]
                (2) scaffold_input['segment_ids']: [batch_size, source_len]
                (3) scaffold_input['offsets_ids']: [batch_size, source_len]
            -decoration_input: a dictionary
                (1) decoration_input['seq_ids']: the un-embeded decoration sequence with the size of [batch_size, target_len]
                (2) decoration_input['segment_ids']: [batch_size, source_len]
                (3) decoration_input['offsets_ids']: [batch_size, source_len]
        return:
            -logits: [batch_size, target_len, n_tokens]
        """
        decoration_embeded = self.embedding(decoration_input['seq_ids']) # [batch_size, target_len-1, d_model]
        position_embedding = PositionEmbedding(
            self.d_model, 
            decoration_input['segment_ids'], 
            decoration_input['offsets_ids'], 
            dropout=self.dropout
        )
        decoration_embeded = position_embedding(decoration_embeded)
        
        scaffold_embeded = self.embedding(scaffold_input['seq_ids']) # [batch_size, source_len, d_model]
        position_embedding = PositionEmbedding(
            self.d_model, 
            scaffold_input['segment_ids'], 
            scaffold_input['offsets_ids'], 
            dropout=self.dropout
        )
        scaffold_embeded = position_embedding(scaffold_embeded) 
        tgt_mask = self._generate_square_subsequent_mask(decoration_embeded.size(1)).to(self.device) # hidden the future information for the decoration_embeded
        decoded = self.decoder(
            decoration_embeded, 
            scaffold_embeded, 
            tgt_mask=tgt_mask
        )
        logits = self.fully_connect_layer(decoded) # [batch_size, target_len-1, n_tokens]
        
        return logits
    
    # Define the main funtion for the training and validation step
    def step(self, batch):
        """
        batch[0]: dictionary of the scaffold, which contains seq_ids, segment_ids, offset_ids, and mask_segment_ids
        batch[1]: dictionary of the decorations, which contains seq_ids, segment_ids, and offset_ids
        """
        scaffold_input, decoration_input = batch[0], batch[1]
        dec = {
            'seq_ids': decoration_input['seq_ids'][:, :-1],
            'segment_ids': decoration_input['segment_ids'][:, :-1],
            'offsets_ids': decoration_input['offsets_ids'][:, :-1]
            
        }
        pred_logits = self.forward(scaffold_input, dec) # [batch_size, seq_len-1, vocab_size]
        pred_logits = pred_logits.view(-1, pred_logits.size(-1)) # Reshape the size to [batch_size*(seq_len-1), vocab_size]
        target = decoration_input['seq_ids'][:, 1:].contiguous().view(-1)
        loss = torch.nn.functional.cross_entropy(pred_logits, target)
            
        return loss
    
    # Perform the training phase
    def training_step(self, batch, batch_idx):
        self.train()
        loss = self.step(batch)
        
        return loss
    
    # Perform the validation phase
    def validation_step(self, batch, batch_idx):
        self.eval()
        loss = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

# ===========================
class FillerSampler():
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        max_len, 
        batch_size, 
        attachment_point_token
    ):
        """
        model: the pretrained model is used for sampling
        max_len: the maximum length of the generated SMILES
        return: a batch size of decorations
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size
        self.attachment_point_token = attachment_point_token
    
    # Sampling a batch of samples by the trained filler
    def decoration_sampler(self, scaffold_input, data=None):
        """
        scaffold_input: type of dictionary, which contains seq_ids, segment_ids, offsets_ids, and mask_segment_ids
        data: 
            -None: generate SMILES for the pretrained filler
            -Not None: generate SMILES for the rollout function
        return: a batch of decorations
        """
        self.model.eval()
        finished = [False] * scaffold_input['seq_ids'].size(0) # Check if each decoration in a batch is finished according to the 'mask_end'
        sample_tensor = torch.zeros((scaffold_input['seq_ids'].size(0), self.max_len), dtype=torch.long)   
        sample_tensor[:, 0] = self.tokenizer.char_to_int[self.tokenizer.mask_start] # Make the first token the mask start char
        with torch.no_grad():
            # Generate SMILES according to the pretrained filler
            if data is None:
                init = 1
            # Generate sub-SMILES according to the rollout function
            else:
                sample_tensor[:, :data.size(1)] = data
                init = data.size(1)
                    
            for i in range(init, self.max_len):
                dec_segment_ids = torch.tensor(np.array(scaffold_input['mask_segment_ids']).reshape(sample_tensor.size(0), 1)).repeat(1, sample_tensor.size(1))
                dec_offsets_ids = torch.arange(0, sample_tensor.size(1)).repeat(sample_tensor.size(0), 1)
                decoration_input = {
                    'seq_ids': sample_tensor,
                    'segment_ids': dec_segment_ids,
                    'offsets_ids': dec_offsets_ids
                } 
                logits = self.model.forward(scaffold_input, decoration_input)[:, i-1] #[batch_size, n_tokens]
                probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze()
                sampled_char = torch.multinomial(probabilities, 1) # [batch_size, 1]
        
                # Stop claculations for the finished sequences
                for idx in range(len(finished)):
                    if finished[idx]:
                        sampled_char[idx] = self.tokenizer.char_to_int[self.tokenizer.mask_end]
                    if sampled_char[idx] == self.tokenizer.char_to_int[self.tokenizer.mask_end]:
                        finished[idx] = True
                sample_tensor[:, i] = sampled_char.squeeze()
                if all(finished):
                    break
        # Decode to scaffold smiles, decoration smiles, and update smiles
        row_smiles = []
        for i in range(scaffold_input['seq_ids'].size(0)):
            dec_smi = self.tokenizer.decode(sample_tensor[i].squeeze().detach().cpu().numpy()).strip('{}_')
            scaff_smi = self.tokenizer.decode(scaffold_input['seq_ids'][i].squeeze().detach().cpu().numpy()).strip('<>_')
            updated_smi = scaff_smi.replace(self.attachment_point_token, dec_smi)
            # Unique SMILES transformation
            updated_unique_smi = canonical_smi(updated_smi)
            # Empty dec_smi will cause NaN problem
            if len(dec_smi) == 0:
                dec_smi = 'c'
            row_smiles.append('{};{};{}'.format(scaff_smi, dec_smi, updated_unique_smi))
        self.model.train()
        
        return row_smiles
        
    # Sampling n samples 
    def multi_sample(
        self, 
        data_loader, 
        filename= None
    ):
        """
        data_loader: data_loader.train_dataloader()
        filename: save the generated samples to the file
        return: completed SMIILES that combined by the scaffolds and decorations
        """
        samples = []
        for batch in data_loader:
            scaffold_input = batch[0]
            row_smiles = self.decoration_sampler(scaffold_input)
            samples.extend(row_smiles)
        
        if filename:
            with open(filename, 'w') as fout:
                for s in samples:
                    fout.write('{}\n'.format(s))
