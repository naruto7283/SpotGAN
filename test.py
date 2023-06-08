import os
import torch
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
import pytorch_lightning
from rdkit.Chem import Draw
from mol_metrics import *
from data_iter import FillerDataLoader
from filler import FillerModel, FillerSampler
warnings.filterwarnings('ignore')

#===========================================================
def evaluation_results(scaffold, test_file, test_property, save_file):
    
    # Read predicted smiles
    data = pd.read_csv(test_file, sep=';', names=['scaffold', 'decoration', 'smiles'])
    total_size = data['smiles'].size
    
    # Valid smiles
    valid_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in data['smiles'] if Chem.MolFromSmiles(smi) is not None and len(smi) != 0]
    valid_smiles = pd.DataFrame(valid_smiles, columns=['smiles'])
    
    # Unique smiles
    unique_smiles = valid_smiles.drop_duplicates(['smiles']) # Delete the same generated smiles from the valid smiles
    
    # Compute mean property
    if test_property == 'DRD2':
        val = batch_DRD2([scaffold])[0]
        vals = batch_DRD2(unique_smiles['smiles'])

    print('\n\n\nScaffold is {}, DRD2 score is {:.3f}'.format(scaffold, val))
    print('Mean {} is {:.3f}\n'.format(test_property, np.mean(vals)))

    # Top-12 
    dic = dict(zip(unique_smiles['smiles'], vals))
    dic = sorted(dic.items(),key=lambda x:x[1],reverse=True)   
    flag = 0
    top_mols= []
    for i in range(len(dic)):
        if flag < 12:
            if dic[i][0] not in top_mols:
                flag += 1
                top_mols.append(Chem.MolFromSmiles(dic[i][0]))
                print(dic[i][0], '\t %.3f' %dic[i][1])    
    img = Draw.MolsToGridImage(top_mols[:], molsPerRow = 3, subImgSize = (1000, 1000), legends = ['' for x in top_mols], returnPNG=False)
    img.save(save_file)

#===========================================================
def test(args):

    NAME = args.scaffold_name
    SIZE = args.test_size 
    MODEL_PATH = args.trained_filler
    PROPERTY = args.test_property
    TEST_PATH = 'results/test/'
    if not os.path.exists(TEST_PATH):
        os.makedirs(TEST_PATH)

    TEST_FILE = TEST_PATH + 'test_dataset.csv'
    SAVE_FILE = TEST_PATH + NAME + '.pdf'

    print('=====================================================================================')
    params = {}
    params['NAME'] = NAME
    params['SIZE'] = SIZE
    params['MODEL_PATH'] = MODEL_PATH
    params['PROPERTY'] = PROPERTY
    params['TEST_FILE'] = TEST_FILE
    params['SAVE_FILE'] = SAVE_FILE
    for param in params:
        string = param + ' ' * (10 - len(param))
        print('{}:   {}'.format(string, params[param]))
    print('=====================================================================================')

    if NAME == 'A1':
        SCAFFOLD = '*C(=O)NCC=CCN1CCNCC1'
    elif NAME == 'A2':
        SCAFFOLD = '*N1CCN(CC=CCNC(=O)Nc2cccc(Cl)c2Cl)CC1'

    elif NAME == 'B1':
        SCAFFOLD = 'CNC1COc2c(*)cccc2C1'    
    elif NAME == 'B2':
        SCAFFOLD = 'CN(*)C1COc2c(cccc2S(=O)(=O)NCCCOC)C1'
    elif NAME == 'B3':
        SCAFFOLD = 'COCCCNS(=O)(=O)c1cccc2c1OCC(N(C*)C(=O)Nc1ccc(F)cc1Cl)C2'

    elif NAME == 'C1':
        SCAFFOLD = 'N1CCN(*)C1=O'
    elif NAME == 'C2':
        SCAFFOLD = '*N1CCN(S(=O)(=O)NCCCC(=O)[O-])C1=O'

    elif NAME == 'D1':
        SCAFFOLD = 'Oc1ccccc1C(=O)NCC1CCCN1*'
    elif NAME == 'D2':
        SCAFFOLD = 'O=C(CN1CCCC1CNC(=O)c1ccccc1O(*))Nc1ccccc1Cl'
    elif NAME == 'D3':
        SCAFFOLD = '*c1cccc(C(=O)NCC2CCCN2CC(=O)Nc2ccccc2Cl)c1ONC(=O)c1ccccc1F'
    elif NAME == 'D4':
        SCAFFOLD = '*c1ccc(CCc2ccccc2O)c(ONC(=O)c2ccccc2F)c1C(=O)NCC1CCCN1CC(=O)Nc1ccccc1Cl'

    elif NAME == 'E1':
        SCAFFOLD = '*c1c(C(=O)NCCN2CCNCC2)cc(Br)c2ccccc12'
    elif NAME == 'E2':
        SCAFFOLD = '*N1CCN(CCNC(=O)c2cc(Br)c3ccccc3c2CC(=O)Nc2ccc(F)cc2)CC1'

    # Generate test dataset
    samples = [SCAFFOLD + ';' + 'c1ccccc1;c1ccccc1' for i in range(SIZE)]

    with open(TEST_FILE, 'w') as fout:
        for s in samples:
            fout.write('{}\n'.format(s))

    # Apply the seed to reproduct the results
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Vocabulary
    tokenizer = Tokenizer()
    tokenizer.build_vocab()

    # Filler train loader
    filler_test_loader = FillerDataLoader(tokenizer, TEST_FILE, SIZE, 100, 64)
    filler_test_loader.setup()
    test_loader = filler_test_loader.train_dataloader()

    # Filler model
    filler = FillerModel(
        tokenizer.n_tokens,
        d_model=256, 
        nhead=4, 
        num_decoder_layers=4, 
        dim_feedforward=100,
        dropout=0.1,
        max_lr=1e-4, 
        epochs=200,
        train_size=SIZE,
        batch_size=64,
        optimizer='RMSprop'
    )

    # Load the trained model
    filler.load_state_dict(torch.load(MODEL_PATH))  
    sampler = FillerSampler(filler, tokenizer, 100, 64, '*')
    sampler.multi_sample(test_loader, TEST_PATH + NAME + '.csv')

    # Print evaluated results  
    evaluation_results(SCAFFOLD, TEST_PATH + NAME + '.csv', PROPERTY, SAVE_FILE)

    
    
    




    
    
    
    
    
    
    
