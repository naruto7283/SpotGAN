import torch
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from mol_metrics import *
import matplotlib.pyplot as plt

# ===========================
def evaluation(
    preprocessed_smiles_file, 
    generated_smile_file, 
    total_data_size, 
    properties=None,
    left_properties=None,
    property_file=None,
    time=None, 
    epoch=None
):
    """
    retrun: validity, uniqueness, novelty
    """
    # Load and make the canonical preprocessed smiles
    preprocessed_smiles = pd.read_csv(
        preprocessed_smiles_file, 
        sep=';', 
        names=['scaffold', 'decorations', 'smiles']
    )
    for i in range(preprocessed_smiles['smiles'].size):
        preprocessed_smiles['smiles'][i] = Chem.MolToSmiles(Chem.MolFromSmiles(preprocessed_smiles['smiles'][i]))
    
    # Save valid smiles
    valid_smiles = {'scaffold': [], 'decorations': [], 'smiles': []}
    with open(generated_smile_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            scaffold, decorations, smiles = line.strip().split(';')
            scaffold = scaffold.strip(' ')
            decorations = decorations.strip(' ')
            smiles = smiles.rstrip('\n').strip(' ')
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and len(smiles) != 0:
                valid_smiles['scaffold'].append(scaffold)
                valid_smiles['decorations'].append(decorations)
                valid_smiles['smiles'].append(Chem.MolToSmiles(mol))
    valid_smiles = pd.DataFrame(valid_smiles)
    len_valid = valid_smiles['smiles'].size
    validity = len_valid / total_data_size
        
    # Save unique smiles
    unique_smiles = valid_smiles.drop_duplicates(['smiles']) # Delete the same generated smiles from the valid smiles
    len_unique = unique_smiles['smiles'].size
    uniqueness = len_unique / len_valid
    
    # Save novel smiles
    novel_smiles = unique_smiles[(~unique_smiles['smiles'].isin(preprocessed_smiles['smiles']))]
    novel_smiles.reset_index(drop=True, inplace=True)
    len_novel = novel_smiles['smiles'].size
    novelty = len_novel / len_unique
    
    # Diversity
    diversity = batch_diversity(novel_smiles['smiles'])
    
    print('\nResults Report:')
    print('*'*80)
    print("Total Mols:   {}".format(total_data_size))
    print("Validity:     {}    ({:.2f}%)".format(len_valid, validity * 100))
    print("Uniqueness:   {}    ({:.2f}%)".format(len_unique, uniqueness * 100))
    print("Novelty:      {}    ({:.2f}%)".format(len_novel, novelty * 100))
    print("Diversity:    {:.2f}".format(diversity))
    print('\n')
    
    # Show top-5 samples from novel smiles
    print('Samples of Novel SMILES:\n')
    if len_novel >= 5:
        print(novel_smiles.sample(5).to_string(index=False))
    else:
        print(novel_smiles.to_string(index=False))
    print('\n')  

    # Compute the property scores of valid smiles 
    if len_valid:
        # Compute decoration scores
        decoration_vals = reward_fn(properties, novel_smiles['decorations'])
        mean_dec, std_dec, min_dec, max_dec = np.mean(decoration_vals), np.std(decoration_vals), np.min(decoration_vals), np.max(decoration_vals)
        print('[{}]: [Mean: {:.3f}   STD: {:.3f}   MIN: {:.3f}   MAX: {:.3f}]'.format('decorations', mean_dec, std_dec, min_dec, max_dec))
        # Compute the generated SMILES scores
        vals = reward_fn(properties, novel_smiles['smiles'])
        mean_s, std_s, min_s, max_s = np.mean(vals), np.std(vals), np.min(vals), np.max(vals)
        print('[{}]: [Mean: {:.3f}   STD: {:.3f}   MIN: {:.3f}   MAX: {:.3f}]'.format(properties, mean_s, std_s, min_s, max_s))
        # Compute other two mean scores
        score_1 = reward_fn(left_properties[0], novel_smiles['smiles'])
        score_1 = np.mean(score_1)
        score_2 = reward_fn(left_properties[1], novel_smiles['smiles'])
        score_2 = np.mean(score_2)
        # Write the property scores into file
        if epoch is not None and time is not None:
            with open(property_file, 'a+') as wf:
                wf.write('{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.5f},{:.5f},{:.5f},{:.5f},{:.3f}\n'.format(epoch, mean_dec, std_dec, min_dec, max_dec, mean_s, std_s, min_s, max_s, score_1, score_2, validity, uniqueness, novelty, diversity, time))
    else:
        print('No valid SMILES generated !')
    print('*'*80)
    print('\n')
    
# ===========================
# Show Top-12 Molecules
def top_mols_show(filename, properties):
    """
    filename: NEGATIVE FILES (generated dataset of SMILES)
    properties: 'druglikeness' or 'solubility' or 'synthesizability'
    """
    mols, scores = [], []
    # Read the generated SMILES data
    generated_smiles = pd.read_csv(
        filename, 
        sep=';', 
        names=['scaffold', 'decorations', 'smiles']
    )
    smiles = generated_smiles['smiles']
    if properties == 'druglikeness':
        scores = batch_druglikeness(smiles)      
    elif properties == 'synthesizability':
        scores = batch_SA(smiles)  
    elif properties == 'solubility':
        scores = batch_solubility(smiles)   
    # Sort the scores
    dic = dict(zip(smiles, scores))
    dic=sorted(dic.items(),key=lambda x:x[1],reverse=True)
    
    flag = 0
    top_mols= []
    for i in range(len(dic)):
        if flag < 12:
            if properties == 'synthesizability':
                if dic[i][0] not in top_mols and dic[i][1] > 0.95 and QED.default(Chem.MolFromSmiles(dic[i][0])) >= 0.5:
                    flag += 1
                    top_mols.append(Chem.MolFromSmiles(dic[i][0]))
                    print(dic[i][0], '\t %.3f' %dic[i][1])
            else:
                 if dic[i][0] not in top_mols:
                    flag += 1
                    top_mols.append(Chem.MolFromSmiles(dic[i][0]))
                    print(dic[i][0], '\t %.3f' %dic[i][1])                    
    return top_mols

# ===========================
# Figure out the distributions
def all_property_scores(
    real_file, 
    gan_file, 
    properties
):
    """
    real_file: real dataset
    gan_file: generated file which contains scaffold, decorations, smiles
    properties: druglikeness, solubility, synthesizability
    """
    # Read real file
    real_lines = open(real_file, 'r').read().splitlines()
    real_lines = [line.split(';')[-1].strip(' ') for line in real_lines]
    # Read gan file
    gan_lines = open(gan_file, 'r').read().splitlines()
    gan_lines = [line.split(';')[-1].strip(' ') for line in gan_lines]
    # Valid SMILES 
    gan_valid = [s for s in gan_lines if Chem.MolFromSmiles(s) and len(s)!=0]
    
    real_scores, gan_scores = [], []
    if properties == 'druglikeness':
        real_scores = batch_druglikeness(real_lines)
        gan_scores = batch_druglikeness(gan_valid)
    elif properties == 'solubility':
        real_scores = batch_solubility(real_lines)
        gan_scores = batch_solubility(gan_valid)
    elif properties == 'synthesizability':
        real_scores = batch_SA(real_lines)
        gan_scores = batch_SA(gan_valid)
    
    return real_scores, gan_scores
    
def draw_distributions(
    real_file, 
    gan_file, 
    save_to_file,
    properties=None
):
    """
    real_file: real dataset
    gan_file: generated file which contains scaffold, decorations, smiles
    properties: druglikeness, solubility, synthesizability
    """
    if properties is not None:
        real_scores, gan_scores = all_property_scores(
            real_file,
            gan_file,
            properties
        )
        # Mean scores
        print('Mean REAL {}: {:.3f}'.format(properties, np.mean(real_scores)))
        print('Mean GAN {}: {:.3f}'.format(properties, np.mean(gan_scores)))
        
        # Plot distribution figures for real dataset and GAN
        plt.subplots(figsize=(12,7))
        # Font size
        plt.xticks( fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel(properties, size=15)
        plt.ylabel('Density', size=15)
        # Set the min and max value of X axis
        plt.xlim(-0.1, 1.1)
    
        sns.kdeplot(real_scores, shade=True, linewidth=1, label='ORIGINAL')
        sns.kdeplot(gan_scores, shade=True, linewidth=1, label='GENERATED')
        plt.legend(loc='upper right', prop={'size': 15})
        plt.savefig(save_to_file)
    else:
        print('Cannot compute the scores because the property is None.')
    
# ===========================
# Policy gradient loss function
def pg_loss(probs, targets, rewards):
    """
    probs: [batch_size * seq_len, vocab_size], logP of the filler output
    targets: [batch_size * seq_len], integers
    rewards: [batch_size, seq_len]
    """
    one_hot = torch.zeros(probs.size(), dtype=torch.bool).to(rewards.device) # [batch_size * seq_len, vocab_size] with all 'False'
    # Set 1 for the token (vocab_size) of each row of one_hot
    one_hot.scatter_(1, targets.data.view(-1, 1), 1) # [batch_size * seq_len, vocab_size] 
    # Select the values in probs according to one_hot
    loss = torch.masked_select(probs, one_hot) # [batch_size * seq_len]
    loss = loss * rewards.contiguous().view(-1) # [batch_size * seq_len]
    loss = - torch.sum(loss)
    
    return loss

# ===========================
# Canonical smiles transformation
def canonical_smi(smi):
    mol = Chem.MolFromSmiles(smi)
    if len(smi) != 0 and mol is not None:
        smi = Chem.MolToSmiles(mol)

    return smi
