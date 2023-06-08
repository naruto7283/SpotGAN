import torch
import numpy as np
from mol_metrics import *
from filler import PositionEmbedding
from pytorch_lightning import LightningModule

# ===========================
# Deep copy the filler model for rollout sampling
class DeepCopiedFiller(LightningModule):

    def __init__(
        self,
        n_tokens, # vocabulary size
        d_model=256,
        nhead=8,
        num_decoder_layers=4,
        dim_feedforward=200,
        dropout=0.1,
        activation='relu',
        max_lr=1e-4,
        epochs=200,
        train_size=10000,
        batch_size=64
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
        self.setup_layers()
    
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
       
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1) 
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        return mask
    
    def forward(self, scaffold_input, decoration_input):
        decoration_embeded = self.embedding(decoration_input['seq_ids']) # [batch_size, target_len-1, d_model]
        position_embedding = PositionEmbedding(
            self.d_model, 
            decoration_input['segment_ids'], 
            decoration_input['offsets_ids'], 
            dropout=self.dropout)
        decoration_embeded = position_embedding(decoration_embeded)
        
        scaffold_embeded = self.embedding(scaffold_input['seq_ids']) # [batch_size, source_len, d_model]
        position_embedding = PositionEmbedding(
            self.d_model, 
            scaffold_input['segment_ids'], 
            scaffold_input['offsets_ids'], 
            dropout=self.dropout)
        scaffold_embeded = position_embedding(scaffold_embeded)
        
        tgt_mask = self._generate_square_subsequent_mask(decoration_embeded.size(1))
        decoded = self.decoder(
            decoration_embeded, 
            scaffold_embeded, 
            tgt_mask = tgt_mask
        )
        logits = self.fully_connect_layer(decoded) # [batch_size, target_len-1, n_tokens]
        
        return logits

# ===========================
# Rollout object
class Rollout(object):
    
    def __init__(
        self, 
        filler, 
        deepcopied_filler, 
        tokenizer,
        attachment_point_token,
        update_rate, 
        whole_smiles,
        device
    ):
        """
        filler: the pretrained filler model which can generate decorations
        deepcopied_filler: the deepcopied filler model 
        tokenizer: Tokenizer object
        update_rate: rate to update the roll_model
        whole_smiles: if true: whole smiles. else: decorations
        """
        # Shallow copy: if mdoel's parameters change, ori_model will change
        self.ori_filler = filler 
        self.deepcopied_filler = deepcopied_filler
        self.tokenizer = tokenizer
        self.attachment_point_token = attachment_point_token
        self.update_rate = update_rate
        self.whole_smiles = whole_smiles
        self.device = device

    def get_reward(
        self, 
        scaffold_input,
        decoration_input,
        rollsampler, 
        rollout_num, 
        dis, 
        dis_lambda=0.5, 
        properties=None
    ):
        """
        scaffold_input: a batch of scaffold input which contain (seq_ids, segment_ids, offset_ids, mask_segment_ids) within the start and end token
        decoration_input: a batch of decorations according to the scaffold input, which contain (seq_ids, segment_ids, offset_ids)
        rollsampler: an object of FillerSampler
        rollout_num: the number of rollout times
        dis: discrimanator model 
        dis_lambda: if 0: Naive RL, elif 1: SeqGAN
        properties: the desired chemical properties 
        """
        np.set_printoptions(threshold=np.inf)
        torch.set_printoptions(threshold=np.inf)
        
        scaffold_smiles = [self.tokenizer.decode(sca.squeeze().detach().cpu().numpy()).strip('<>_') for sca in scaffold_input['seq_ids']]
        decoration_smiles = [self.tokenizer.decode(dec.squeeze().detach().cpu().numpy()).strip('{}_') for dec in decoration_input['seq_ids']]
        if self.whole_smiles:
            # Generated SMILES
            sample_smiles = [scaffold_smiles[idx].replace(self.attachment_point_token, decoration_smiles[idx]) for idx in range(len(scaffold_smiles))]
        else:
            # Generated decorations
            sample_smiles = decoration_smiles
        batch_size, seq_len = decoration_input['seq_ids'].size()
        dis.to(self.device)
        # Inactivate the dropout layer
        dis.eval() 
        rewards = []
        # Start from the second letter (after the start token and the first action)
        init = 2 
        
        for i in range(rollout_num):
            # Delete the traversed SMILES
            already = [] 
            # Generate SMILES based on the given sub-SMILES
            for given_num in range(init, seq_len):
                generated_samples = rollsampler.decoration_sampler(scaffold_input, decoration_input['seq_ids'][:, :given_num]) # batch_size
                if self.whole_smiles:
                    # Generated SMILES are fed into discriminator and reward_fn 
                    generated_smiles = [sam.split(';')[-1].strip() for sam in generated_samples]
                else:
                    # Generated decorations are fed into discriminator and reward_fn 
                    generated_smiles = [sam.split(';')[-2].strip() for sam in generated_samples]
                generated_smiles_encoded = [torch.tensor(self.tokenizer.scaffold_encode(smi))[1:-1] for smi in generated_smiles]
                generated_smiles_paded = torch.nn.utils.rnn.pad_sequence(
                    generated_smiles_encoded, 
                    batch_first=True, 
                    padding_value=self.tokenizer.char_to_int[self.tokenizer.pad]
                ).squeeze().to(self.device) # [batch_size, max_len]
                gind = np.array(range(generated_smiles_paded.size(0))) # batch_size
                dis_pred = dis.forward(generated_smiles_paded) # [batch_size, 2]
                dis_pred = torch.nn.functional.softmax(dis_pred, dim=1) # [batch_size, 2]
                # Probability of real class
                dis_pred = dis_pred.data[:, 1].cpu().numpy() 
                dis_pred = dis_lambda * dis_pred
                # Delete sequences that are already finished, and add their rewards
                for k, r in reversed(already):
                    del  generated_smiles[k]
                    gind = np.delete(gind, k, 0)
                    dis_pred[k] += (1 - dis_lambda) * r
                # If there are still seqs, calculate rewards
                if len(generated_smiles): # batch_size
                    vals = reward_fn(properties, generated_smiles)
                    if self.whole_smiles:
                        pct_unique = len(list(set(generated_smiles))) / float(len(generated_smiles))
                        weights = np.array([pct_unique / float(generated_smiles.count(sm)) for sm in generated_smiles])
                        rew = vals * weights
                    else:
                        rew = np.array(vals)
                # Add the just calculated rewards
                for k, r in zip(gind, rew):
                    dis_pred[k] += (1 - dis_lambda) * r
                # Choose the seqs finished in the last iteration
                for j, k in enumerate(gind): # k: real idx of gind
                    if decoration_input['seq_ids'][k][given_num-1] == self.tokenizer.char_to_int[self.tokenizer.mask_end]:
                        already.append((k, rew[j]))                            
                already = sorted(already, key = lambda el: el[0]) 
                if i == 0:
                    rewards.append(dis_pred) # [give_num-1, batch_size]
                else:
                    rewards[given_num - init] += dis_pred # [seq_len-1, batch_size]
            # For the last token 
            last_encoded = [torch.tensor(self.tokenizer.scaffold_encode(s))[1:-1] for s in sample_smiles]
            last_paded = torch.nn.utils.rnn.pad_sequence(
                last_encoded,
                batch_first=True, 
                padding_value=self.tokenizer.char_to_int[self.tokenizer.pad]
            ).squeeze().to(self.device) # [batch_size, max_len]
            dis_pred = dis.forward(last_paded) # [batch_size, 2]
            dis_pred = torch.nn.functional.softmax(dis_pred, dim=1).cpu()
            dis_pred = dis_pred.data[:, 1].numpy()
            dis_pred = dis_lambda * dis_pred
            vals = reward_fn(properties, sample_smiles)
            if self.whole_smiles:
                # Compute rewards of SMILES
                pct_unique = len(list(set(sample_smiles))) / float(len(sample_smiles))
                weights = np.array([pct_unique / float(sample_smiles.count(s)) for s in sample_smiles])     
                rew = vals * weights
            else:
                # Compute rewards of decorations
                rew = np.array(vals)
            dis_pred += (1 - dis_lambda) * rew
            if i == 0:
                rewards.append(dis_pred)
            else:
                rewards[-1] += dis_pred

        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num) # [batch_size, seq_len-1]
        rewards = rewards - np.mean(rewards)
        # Activate the dropout layer
        dis.train() 

        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_filler.named_parameters():
            dic[name] = param.data
        for name, param in self.deepcopied_filler.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]

