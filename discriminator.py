import math
import torch
import numpy as np
from pytorch_lightning import LightningModule

# ===========================
# Define the position embedding manually
class PositionalEmbedding(torch.nn.Module):
    
    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_len=500
    ):
        """
        d_model: the embedded dimension
        max_len: the maximum length of sequences
        """
        super(PositionalEmbedding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1e4) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # PosEncoder(pos, 2i) = sin(pos/10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, embed):
        """
        embed: the embeded sequence with the tensor size of [batch_size, seq_len, d_model]
        return: the embedding with the tensor size of [batch_size, seq_len, d_model]
        """
        # [batch_size, seq_len, d_model] + [1, seq_len, d_model] 
        embed = embed + self.pe[:, :embed.size(1)]
        embed = self.dropout(embed)
        
        return embed

# ===========================
# Definition of the discriminator model
class DiscriminatorModel(LightningModule):
    
    def __init__(
        self,
        n_tokens, # Vocabulary size
        d_model=256,
        nhead=4,
        num_encoder_layers=4,
        dim_feedforward=200,
        dropout=0.1,
        max_lr=1e-5,
        epochs=10,
        pad_token=0,
        train_size=20000,
        batch_size=64,
        dis_wgan=False,
        minibatch=False   
    ):
        super().__init__()
        assert d_model % nhead == 0,  "nheads must divide evenly into d_model" 
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_lr = max_lr
        self.epochs = epochs
        self.pad_token = pad_token
        self.train_size = train_size
        self.batch_size = batch_size
        self.dis_wgan = dis_wgan
        self.minibatch = minibatch
        self.setup_layers()
        self.steps_per_epoch = math.ceil(self.train_size/self.batch_size)
        
    # Initialize parameters with truncated normal distribution for the classifer
    def truncated_normal_(
        self, 
        tensor, 
        mean=0,
        std=0.1
    ):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            
            return tensor
        
   # Apply WGAN loss to alleviate the mode-collapse problem
    def wgan_loss(self, outputs, labels):
        """
        outputs: the digit outputs of discriminator forward function with size [batch_size, 2]
        labels: labels of the discriminator with size [batch_size]
        """
        assert len(labels.shape) == 1
        assert outputs.shape[0] == labels.shape[0]
        # partation the outputs according to the label 0 and 1
        neg, pos = [outputs[labels == i] for i in range(2)]
        w_loss = torch.abs(torch.sum(neg) / (neg.shape[0] + 1e-10) - torch.sum(pos) / (pos.shape[0] + 1e-10))
        
        return w_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.max_lr, 
            total_steps=None, 
            epochs=self.epochs, 
            steps_per_epoch=self.steps_per_epoch,
            pct_start=6/self.epochs, 
            anneal_strategy='cos', 
            cycle_momentum=True, 
            base_momentum=0.85, 
            max_momentum=0.95,
            div_factor=1e3, 
            final_div_factor=1e3, 
            last_epoch=-1
        )        
        scheduler = {"scheduler": scheduler, "interval" : "step" }
        
        return [optimizer], [scheduler]     
        
    def setup_layers(self):
        self.embedding = torch.nn.Embedding(self.n_tokens, self.d_model)
        self.positional_encoder = PositionalEmbedding(self.d_model, dropout=self.dropout)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            self.d_model, 
            self.nhead, 
            self.dim_feedforward, 
            self.dropout
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, self.num_encoder_layers)

        if self.minibatch:
            self.classifier = torch.nn.Linear(self.d_model+1, 2)
        else:
            self.classifier = torch.nn.Linear(self.d_model, 2)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def padding_mask(self, src): # src:[batch_size, maxlength]
        return src == self.pad_token # [batch_size, maxlength]
        
    # Omit the effect of padding token
    def masked_mean(self, encoded, mask):
        """
        encoded: output of TransformerEncoder with size [batch_size, maxlength, d_model]
        mask: output of padding_mask with size [batch_size, maxlength]: if pad: True, else False
        return: mean of the encoded according to the non-zero/True mask [batch_size, d_model]
        """
        non_mask = mask.unsqueeze(-1) == False # [batch_size, maxlength, 1] if Pad: 0, else 1
        masked_encoded = encoded * non_mask # [batch_size, maxlength, d_model]
        ave = masked_encoded.sum(dim=1) / non_mask.sum(dim=1) # [batch_size, d_model]
        
        return ave
        
    # Apply mini-batch discrimination to alleviate the mode-collapse problem
    def minibatch_std(self, x):
        """
        x: output of the middle layer of Discriminator with size [batch_size, d_model]
        return: contains the mean of the std information of x
        """
        size = list(x.size())
        size[1] = 1
        # Compute std according to the batch_size direction
        std = torch.std(x, dim=0)
        mean = torch.mean(std)
        return torch.cat((x, mean.repeat(size)), dim=1)
    
    def forward(self, features): 
        """
        features: [batch_size, maxlength]
        """
        paded_mask = self.padding_mask(features)
        embedded = self.embedding(features) * math.sqrt(self.d_model) #[batch_size, maxlength, d_model]
        positional_encoded = self.positional_encoder(embedded) #[batch_size, maxlength, d_model]
        encoded = self.encoder(positional_encoded) # [batch_size, maxlength, d_model]
        masked_out = self.masked_mean(encoded, paded_mask) # [batch_size, d_model]
        # If true: apply mini-batch discriminator
        if self.minibatch:
            masked_out = self.minibatch_std(masked_out)
        # If true: apply WGAN
        if self.dis_wgan:
            weight_loss = torch.sum(self.classifier.weight**2) / 2.
            bias_loss = torch.sum(self.classifier.bias**2) / 2.
            self.l2_loss = weight_loss + bias_loss
        out = self.classifier(masked_out) #[batch_size, 2]

        return out

    def step(self, batch):
        inputs, labels = batch # inputs:[batch_size, maxlength], labels: [batch_size]
        outputs = self.forward(inputs) #[batch_size, 2]
        if self.dis_wgan:
            # Compute WGAN loss
            w_loss = self.wgan_loss(outputs, labels)
            loss = w_loss + self.l2_loss * 0.2 
        else:
            # Compute cross-entropy loss for GAN
            loss = self.criterion(outputs, labels)
        # Compute accuracy for the classifier 
        pred = outputs.data.max(1)[1] # Indices of max elements
        acc = pred.eq(labels.data).cpu().sum() / len(labels)
        return loss, acc
    
    def training_step(self, batch, batch_idx):
        self.train()
        loss, acc = self.step(batch)

        return loss
    
    def validation_step(self, batch, batch_idx):
        self.eval()
        loss, acc = self.step(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss