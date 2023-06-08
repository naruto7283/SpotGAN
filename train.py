import os
import time
import copy
import torch
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
import pytorch_lightning
from rdkit.Chem import Draw

from utils import *
from mol_metrics import Tokenizer
from discriminator import DiscriminatorModel
from data_preprocess import DataPreprocessor
from filler import FillerModel, FillerSampler
from rollout import DeepCopiedFiller, Rollout
from data_iter import FillerDataLoader, DisDataLoader

warnings.filterwarnings('ignore')


def train(args):

    # ===========================
    # Apply the seed to reproduct the results
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # ===========================
    # Data Preprocess
    PREPROCESS = args.preprocess
    DEC_MIN_LEN = args.dec_min_len
    DATASET_NAME = args.dataset_name
    ATTACHMENT_POINT_TOKEN = args.attachment_point_token
    SMARTS = Chem.MolFromSmarts(args.smarts)
    ORIGINAL_DATASET = args.original_dataset
    PREPROCESSED_FILE = args.preprocessed_file
    POSITIVE_FILE = args.positive_file

    # ===========================
    # Filler Model
    FILLER_PRETRAIN = args.filler_pretrain
    FILLER_EPOCHS = args.filler_epochs
    FILLER_GENERATED_SIZE = args.filler_generated_size
    DECORATION_MAX_LEN = args.decoration_max_len
    BATCH_SIZE = args.batch_size
    FILLER_NUM_DECODER_LAYERS = args.filler_num_decoder_layers
    FILLER_DIM_FEEDFORWARD = args.filler_dim_feedforward
    FILLER_D_MODEL = args.filler_d_model
    FILLER_NUM_HEADS = args.filler_num_heads
    FILLER_MAX_LR = args.filler_max_lr
    FILLER_DROPOUT = args.filler_dropout
    FILLER_OPTIMIZER = args.filler_optimizer
    NEGATIVE_FILE = args.negative_file
    FILLER_PRETRAINED_MODEL = args.filler_pretrained_model

    # ===========================
    # Discriminator Model
    DIS_PRETRAIN = args.dis_pretrain
    DIS_WGAN = args.dis_wgan
    DIS_MINIBATCH = args.dis_minibatch
    DIS_PRE_EPOCHS = args.dis_pre_epochs
    DIS_MAX_LR = args.dis_max_lr
    DIS_NUM_ENCODER_LAYERS = args.dis_num_encoder_layers
    DIS_D_MODEL = args.dis_d_model
    DIS_NUM_HEADS = args.dis_num_heads
    DIS_DIM_FEEDFORWARD = args.dis_dim_feedforward
    DIS_DROPOUT = args.dis_dropout
    DIS_PRETRAINED_MODEL = args.dis_pretrained_model

    # ===========================
    # Adversarial
    ADVERSARIAL_TRAIN = args.adversarial_train
    WHOLE_SMILES = args.whole_smiles
    ADV_EPOCHS = args.adv_epochs
    DIS_LAMBDA = args.dis_lambda
    ADV_LR = args.adv_lr
    PROPERTIES = args.properties
    LEFT_PROPERTIES = args.left_properties
    ROLL_NUM = args.roll_num
    MODEL_NAME = args.model_name
    D_STEPS = args.d_steps
    D_EPOCHS = args.d_epochs
    UPDATE_RATE = args.update_rate

    # ===========================
    # Save Models
    SAVE_NAME = args.save_name
    PATHS = args.paths
    if not os.path.exists(PATHS):
        os.makedirs(PATHS)
    # Save the change of property (rewards) at each adversarial training epoch
    PROPERTY_FILE = args.property_file
    if ADVERSARIAL_TRAIN:
        with open(PROPERTY_FILE, 'a+') as wf:
            wf.truncate(0)
            wf.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format('Epoch', 'Mean_dec', 'Std_dec', 'Min_dec', 'Max_dec', 'Mean_SMILES', 'Std_SMILES', 'Min_SMILES', 'Max_SMILES', LEFT_PROPERTIES[0], LEFT_PROPERTIES[1], 'Validity', 'Uniqueness', 'Novelty', 'Diversity', 'Time'))
    # Save the generator at each adversarial training epoch
    ScaffGAN_FILLER_MODEL = args.scaffgan_filler_model
    ScaffGAN_DIS_MODEL = args.scaffgan_dis_model
    TOP_12_GAN = args.top_12_gan
    TOP_12_WGAN = args.top_12_wgan
    DRAW_DISTRIBUTIONS = args.draw_distributions

    # ===========================
    # GPU 
    GPUS = args.gpus
    DEVICE = args.device

    # ===========================
    # Vocabulary
    print('\n\n\nVocabulary Information:')
    print('='*80) 
    tokenizer = Tokenizer()
    tokenizer.build_vocab()
    print(tokenizer.char_to_int)

    SMARTS = Chem.MolFromSmarts(args.smarts)

    # Save all hyperparameters
    with open(PATHS + '/hyperparameters.csv', 'a+') as hp:
        # Clean the hyperparameters file
        hp.truncate(0)

        params = {}
        print('\n\nHyperparameter Information:')
        print('=====================================================================================')
        params['ORIGINAL_DATASET'] = ORIGINAL_DATASET
        params['PREPROCESSED_FILE'] = PREPROCESSED_FILE
        params['POSITIVE_FILE'] = POSITIVE_FILE
        params['NEGATIVE_FILE'] = NEGATIVE_FILE
        params['FILLER_PRETRAINED_MODEL'] = FILLER_PRETRAINED_MODEL
        params['DIS_PRETRAINED_MODEL'] = DIS_PRETRAINED_MODEL 
        params['PROPERTY_FILE'] = PROPERTY_FILE
        params['TOP_12_GAN'] = TOP_12_GAN
        params['TOP_12_WGAN'] = TOP_12_WGAN
        params['DRAW_DISTRIBUTIONS'] = DRAW_DISTRIBUTIONS
        params['MODEL_NAME'] = MODEL_NAME
        for param in params:
            string = param + ' ' * (25 - len(param))
            print('{}:   {}'.format(string, params[param]))
            hp.write('{}\t{}\n'.format(str(param), str(params[param])))
        print('\n')
        
        params = {}
        params['PREPROCESS'] = PREPROCESS
        params['DATASET_NAME'] = DATASET_NAME
        params['SMARTS'] = '[*]!@-[*]'
        params['ATTACHMENT_POINT_TOKEN'] = ATTACHMENT_POINT_TOKEN
        params['DEC_MIN_LEN'] = DEC_MIN_LEN
        params['BATCH_SIZE'] = BATCH_SIZE
        params['DECORATION_MAX_LEN'] = DECORATION_MAX_LEN
        params['VOCAB_SIZE'] = len(tokenizer.char_to_int)
        params['DEVICE'] = DEVICE
        params['GPUS'] = GPUS
        for param in params:
            string = param + ' ' * (25 - len(param))
            print('{}:   {}'.format(string, params[param]))
            hp.write('{}\t{}\n'.format(str(param), str(params[param])))
        print('\n')

        params = {}
        params['FILLER_PRETRAIN'] = FILLER_PRETRAIN
        params['FILLER_MAX_LR'] = FILLER_MAX_LR
        params['FILLER_GENERATED_SIZE'] = FILLER_GENERATED_SIZE
        params['FILLER_NUM_DECODER_LAYERS'] = FILLER_NUM_DECODER_LAYERS
        params['FILLER_DIM_FEEDFORWARD'] = FILLER_DIM_FEEDFORWARD
        params['FILLER_D_MODEL'] = FILLER_D_MODEL
        params['FILLER_NUM_HEADS'] = FILLER_NUM_HEADS
        params['FILLER_DROPOUT'] = FILLER_DROPOUT
        params['FILLER_OPTIMIZER'] = FILLER_OPTIMIZER
        params['FILLER_EPOCHS'] = FILLER_EPOCHS
        for param in params:
            string = param + ' ' * (25 - len(param))
            print('{}:   {}'.format(string, params[param]))
            hp.write('{}\t{}\n'.format(str(param), str(params[param])))
        print('\n')

        params = {}
        params['DIS_PRETRAIN'] = DIS_PRETRAIN
        params['DIS_WGAN'] = DIS_WGAN
        params['DIS_MINIBATCH'] = DIS_MINIBATCH
        params['DIS_NUM_ENCODER_LAYERS'] = DIS_NUM_ENCODER_LAYERS
        params['DIS_D_MODEL'] = DIS_D_MODEL
        params['DIS_NUM_HEADS'] = DIS_NUM_HEADS
        params['DIS_MAX_LR'] = DIS_MAX_LR
        params['DIS_DIM_FEEDFORWARD'] = DIS_DIM_FEEDFORWARD
        params['DIS_DROPOUT'] = DIS_DROPOUT
        params['DIS_PRE_EPOCHS'] = DIS_PRE_EPOCHS
        for param in params:
            string = param + ' ' * (25 - len(param))
            print('{}:   {}'.format(string, params[param]))
            hp.write('{}\t{}\n'.format(str(param), str(params[param])))
        print('\n')

        params = {}
        params['ADVERSARIAL_TRAIN'] = ADVERSARIAL_TRAIN
        params['WHOLE_SMILES'] = WHOLE_SMILES
        params['PROPERTIES'] = PROPERTIES
        params['DIS_LAMBDA'] = DIS_LAMBDA
        params['ADV_LR'] = ADV_LR
        params['UPDATE_RATE'] = UPDATE_RATE
        params['D_STEPS'] = D_STEPS
        params['D_EPOCHS'] = D_EPOCHS
        params['ROLL_NUM'] = ROLL_NUM
        params['ADV_EPOCHS'] = ADV_EPOCHS
        for param in params:
            string = param + ' ' * (25 - len(param))
            print('{}:   {}'.format(string, params[param]))
            hp.write('{}\t{}\n'.format(str(param), str(params[param])))
        print('=====================================================================================')

    # ===========================
    start_time = time.time()
    print('\nStart time : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
    
    # ===========================
    # Data preprocessing
    if PREPROCESS:
        preprocessor = DataPreprocessor(
            ORIGINAL_DATASET, 
            PREPROCESSED_FILE, 
            SMARTS,
            ATTACHMENT_POINT_TOKEN,
            DEC_MIN_LEN,
            1
        )
        smiles = preprocessor.read_smiles_from_file()
        preprocessor.write_smiles_to_file(smiles)
        # Gnereate FILLER_TRAIN_SIE dataset
        smiles = pd.read_csv(PREPROCESSED_FILE)    
        smiles = smiles.sample(FILLER_GENERATED_SIZE)
        smiles.to_csv(POSITIVE_FILE, header=False, index=False)
    else:
        print('\n{} training data are saved in : {}.\n'.format(FILLER_GENERATED_SIZE, POSITIVE_FILE))
    
    # ===========================
    # Filler train loader
    filler_train_loader = FillerDataLoader(
        tokenizer,
        POSITIVE_FILE, 
        FILLER_GENERATED_SIZE, 
        200,
        BATCH_SIZE
    )
    filler_train_loader.setup()
    train_loader = filler_train_loader.train_dataloader()

    # Filler model
    filler = FillerModel(
        tokenizer.n_tokens,
        d_model=FILLER_D_MODEL, 
        nhead=FILLER_NUM_HEADS, 
        num_decoder_layers=FILLER_NUM_DECODER_LAYERS, 
        dim_feedforward=FILLER_DIM_FEEDFORWARD,
        dropout=FILLER_DROPOUT,
        max_lr=FILLER_MAX_LR, 
        epochs=FILLER_EPOCHS,
        train_size=FILLER_GENERATED_SIZE,
        batch_size=BATCH_SIZE,
        optimizer=FILLER_OPTIMIZER
    )
    filler_trainer = pytorch_lightning.Trainer(
        max_epochs=FILLER_EPOCHS,
        gpus=GPUS, 
        enable_model_summary=False,
        gradient_clip_val=0.5,
        progress_bar_refresh_rate=5
    )
    
    # Pretrain the filler
    if FILLER_PRETRAIN:
        print('\nPretraining Filler...\n')
        filler_trainer.fit(filler, filler_train_loader)
        current_time = (time.time() - start_time) / 3600.
        print('\nFiller Pretrain Time:\033[1;35m {:.2f}\033[0m hours'.format(current_time))
        torch.save(filler.state_dict(), FILLER_PRETRAINED_MODEL)
    else:
        filler.load_state_dict(torch.load(FILLER_PRETRAINED_MODEL))
        print('\nPretrained filler has been loaded: {}'.format(FILLER_PRETRAINED_MODEL))
    #"""    
    # Generate samples
    print('\nGenerating {} samples...'.format(FILLER_GENERATED_SIZE))
    filler_sampler =  FillerSampler(
        filler, 
        tokenizer, 
        DECORATION_MAX_LEN, 
        BATCH_SIZE, 
        ATTACHMENT_POINT_TOKEN
    )
    filler_sampler.multi_sample(
        train_loader,
        NEGATIVE_FILE
    )
    if FILLER_PRETRAIN:
        evaluation(
            POSITIVE_FILE,
            NEGATIVE_FILE, 
            FILLER_GENERATED_SIZE, 
            PROPERTIES,
            LEFT_PROPERTIES,
            PROPERTY_FILE,
            current_time,
            0
        )
    else:
        evaluation(
            POSITIVE_FILE,
            NEGATIVE_FILE, 
            FILLER_GENERATED_SIZE, 
            PROPERTIES,
            LEFT_PROPERTIES
        )

    #"""
    # ===========================
    # Discriminator train loader
    DIS_TRAIN_SIZE = FILLER_GENERATED_SIZE * 2
    dis_data_loader = DisDataLoader(
        POSITIVE_FILE,
        NEGATIVE_FILE, 
        tokenizer, 
        BATCH_SIZE
    )
    dis_data_loader.setup()
    
    # Discriminator model
    dis = DiscriminatorModel(
        n_tokens=tokenizer.n_tokens, 
        d_model=DIS_D_MODEL,
        nhead=DIS_NUM_HEADS, 
        num_encoder_layers=DIS_NUM_ENCODER_LAYERS,
        dim_feedforward=DIS_DIM_FEEDFORWARD,
        dropout=DIS_DROPOUT,  
        max_lr=DIS_MAX_LR,
        epochs=DIS_PRE_EPOCHS,
        pad_token=tokenizer.char_to_int[tokenizer.pad],
        train_size=DIS_TRAIN_SIZE,
        batch_size=BATCH_SIZE,
        dis_wgan=DIS_WGAN,
        minibatch=DIS_MINIBATCH
    )
    dis_trainer = pytorch_lightning.Trainer(
        max_epochs=DIS_PRE_EPOCHS, 
        gpus=GPUS, 
        weights_summary=None,
        gradient_clip_val=1.0,
        gradient_clip_algorithm='value'
    )
    
    # Pretrain the discriminator
    if DIS_PRETRAIN:
        print('\nPretraining Discriminator...')
        dis_trainer.fit(dis, dis_data_loader)
        print('Discriminator Pretrain Time:\033[1;35m {:.2f}\033[0m hours'.format((time.time() - start_time) / 3600.))
        torch.save(dis.state_dict(), DIS_PRETRAINED_MODEL)
    else:
        if DIS_LAMBDA:
            dis.load_state_dict(torch.load(DIS_PRETRAINED_MODEL))
            print('\nPretrained discriminator has been loaded: {}'.format(DIS_PRETRAINED_MODEL))
        
    # ===========================
    #Adversarial Training
    rollout_filler = DeepCopiedFiller(
        tokenizer.n_tokens,
        d_model=FILLER_D_MODEL, 
        nhead=FILLER_NUM_HEADS, 
        num_decoder_layers=FILLER_NUM_DECODER_LAYERS, 
        dim_feedforward=FILLER_DIM_FEEDFORWARD,
        dropout=FILLER_DROPOUT,
        max_lr=FILLER_MAX_LR, 
        epochs=FILLER_EPOCHS,
        train_size=FILLER_GENERATED_SIZE,
        batch_size=BATCH_SIZE
    )
    dic = {}
    # Deepcopy the filler model  as rollout network
    filler.requires_grad = False
    for name, param in filler.named_parameters():
        dic[name] = copy.deepcopy(param.data) # # Deepcopy: if model's parameter changes, parameters in own_model do not change 
    for name, param in rollout_filler.named_parameters():
        param.data = dic[name]
    filler.requires_grad = True
    adv_trainer = pytorch_lightning.Trainer(
        max_epochs=D_EPOCHS,
        gpus=GPUS, 
        weights_summary=None,
        progress_bar_refresh_rate=0
    )
    pg_optimizer = torch.optim.Adam(
        params = filler.parameters(), 
        lr = ADV_LR
    )
    rollout = Rollout(
        filler, 
        rollout_filler, 
        tokenizer, 
        ATTACHMENT_POINT_TOKEN, 
        UPDATE_RATE, 
        WHOLE_SMILES,
        DEVICE
    )
    # Update the negative file
    updated_filler_loader = FillerDataLoader(
        tokenizer,
        NEGATIVE_FILE, # Generate scaffold_input and decoration_input from negative file
        FILLER_GENERATED_SIZE, 
        200,
        BATCH_SIZE
    )
    updated_filler_loader.setup()       
    negative_loader = updated_filler_loader.train_dataloader()
    negative_iter = iter(negative_loader)

    # Adversarial training
    if ADVERSARIAL_TRAIN:
        print('\nAdversarial Training...')
        for epoch in range(ADV_EPOCHS):
            rollsampler =  FillerSampler(
                rollout.deepcopied_filler, 
                tokenizer, 
                DECORATION_MAX_LEN, 
                BATCH_SIZE, 
                ATTACHMENT_POINT_TOKEN
            )
            try:
                batch_scaffold_input, batch_decoration_input = next(negative_iter)
            except StopIteration:
                negative_iter = iter(negative_loader)
                batch_scaffold_input, batch_decoration_input = next(negative_iter)
            # Take ONLY a batch of generated samples from the generated file
            dec_filler_input = {
                'seq_ids': batch_decoration_input['seq_ids'][:, :-1],
                'segment_ids': batch_decoration_input['segment_ids'][:, :-1],
                'offsets_ids': batch_decoration_input['offsets_ids'][:, :-1]
            }
            dec_logits = filler.forward(batch_scaffold_input, dec_filler_input) # [batch_size, target_len-1, n_tokens]
            dec_logits = dec_logits.contiguous().view(-1, dec_logits.size(-1)) # [batch_size*(target_len-1), n_tokens]
            dec_logits = torch.nn.functional.log_softmax(dec_logits, dim=1).to(DEVICE)
            dec_targets = batch_decoration_input['seq_ids'][:, 1:].contiguous().view(-1).to(DEVICE) # [batch_size * (target_len-1)]
            # Calculate the rewards of each token in a batch 
            rewards = rollout.get_reward(
                batch_scaffold_input,
                batch_decoration_input,
                rollsampler,
                ROLL_NUM,
                dis,
                DIS_LAMBDA,
                PROPERTIES
            ) # [batch_size, target_len-1]
            rewards = torch.tensor(rewards).to(DEVICE)
            # Compute policy gradient loss
            loss = pg_loss(dec_logits, dec_targets, rewards)
            print('\n\n\n\033[1;35mEpoch {}\033[0m / {}, PG_Loss: {:.3f}'.format(epoch+1, ADV_EPOCHS, loss))  
            pg_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filler.parameters(), 5, norm_type=2)
            pg_optimizer.step()   
            # Update parameters of the rollout network
            rollout.update_params()
            # Save models
            torch.save(filler.state_dict(), PATHS + '/Epoch_' + str(epoch+1) + '_filler.pkl')
            if DIS_LAMBDA:
                torch.save(dis.state_dict(), PATHS + '/Epoch_' + str(epoch+1) + '_dis.pkl')
            # Generate Samples
            filler_sampler =  FillerSampler(
                filler, 
                tokenizer, 
                DECORATION_MAX_LEN, 
                BATCH_SIZE, 
                ATTACHMENT_POINT_TOKEN
            )

            filler_sampler.multi_sample(
                train_loader, 
                NEGATIVE_FILE
            )

            current_time = (time.time() - start_time) / 3600.
            print('\nTotal Computational Time: \033[1;35m {:.2f} \033[0m hours.'.format(current_time))
            evaluation(
                POSITIVE_FILE,
                NEGATIVE_FILE, 
                FILLER_GENERATED_SIZE, 
                PROPERTIES,
                LEFT_PROPERTIES,
                PROPERTY_FILE,
                current_time,
                epoch+1
            )
            # Check if train discriminator
            if DIS_LAMBDA:
                if (epoch+1) % D_STEPS == 0:
                    dis_data_loader.setup()
                    adv_trainer.fit(dis, dis_data_loader)
    else:
        # Load the trained model
        if not os.path.exists(ScaffGAN_FILLER_MODEL):
            print('\nFiller path does NOT exist: ' + ScaffGAN_FILLER_MODEL)
            return
        else:
            print('\nLoad filler: {}'.format(ScaffGAN_FILLER_MODEL))
            filler.load_state_dict(torch.load(ScaffGAN_FILLER_MODEL))
            print('\nGenerating {} samples...'.format(FILLER_GENERATED_SIZE))  
            sampler = FillerSampler(
                filler, 
                tokenizer, 
                DECORATION_MAX_LEN, 
                BATCH_SIZE, 
                ATTACHMENT_POINT_TOKEN
            )
            sampler.multi_sample(
                train_loader, 
                NEGATIVE_FILE
            )
            evaluation(
                POSITIVE_FILE,
                NEGATIVE_FILE, 
                FILLER_GENERATED_SIZE, 
                PROPERTIES,
                LEFT_PROPERTIES
            )

        if DIS_LAMBDA:
            if not os.path.exists(ScaffGAN_DIS_MODEL):
                print('\nDiscriminator path does NOT exist: ' + ScaffGAN_DIS_MODEL)
                return
            else:
                print('\nLoad discriminator: {}'.format(ScaffGAN_DIS_MODEL))
                dis.load_state_dict(torch.load(ScaffGAN_DIS_MODEL))

    # Show Top-12 molecules
    if not os.path.isfile(NEGATIVE_FILE):
        print('\nGenerated dataset does NOT exist!')
    else:
        print('\nTop-12 Molecules of [{}]:'.format(PROPERTIES))
        print('\n')
        top_mols = top_mols_show(NEGATIVE_FILE, PROPERTIES)
        print('*'*80)
        # Save Top-12 figure
        img = Draw.MolsToGridImage(top_mols[:], molsPerRow = 3, subImgSize = (1000, 1000), legends = ['' for x in top_mols], returnPNG=False)
        if DIS_WGAN:
            img.save(TOP_12_WGAN)
            print('\nTop-12 figure is saved in : {}'.format(TOP_12_WGAN))
        else:
            img.save(TOP_12_GAN)
            print('\nTop-12 figure is saved in : {}'.format(TOP_12_GAN))
        
    # Figure out distributions
    print('\nDraw distributions of [{}]:'.format(PROPERTIES))
    draw_distributions(
        POSITIVE_FILE,
        NEGATIVE_FILE,
        DRAW_DISTRIBUTIONS,
        PROPERTIES
    )
    print('\nDrawed distributions are saved in: {}'.format(DRAW_DISTRIBUTIONS))  
                    
    
    




    
    
    
    
    
    
    
