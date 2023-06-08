import os
import torch
import argparse
from test import test
from rdkit import Chem
from train import train

# Default settings
parser = argparse.ArgumentParser()

# ===========================
# Data Preprocess
parser.add_argument('--preprocess', action='store_true', help='whether upreprocessing the original dataset')
parser.add_argument('--dec_min_len', type=int, default=3, help='minimum length of the decorations')
parser.add_argument('--attachment_point_token', type=str, default='*', help='define the attachment point')
parser.add_argument('--smarts', type=str, default='[*]!@-[*]', help='define the SMARTS regular expressions for molecular structures')
parser.add_argument('--dataset_name', type=str, default='QM9_10k', help='use QM9_10k or ZINC_10k dataset')

# ===========================
# Filler Model
parser.add_argument('--filler_pretrain', action='store_true', help='whether pretraining the filler model')
parser.add_argument('--filler_epochs', type=int, default=100, help='define the pretraining epochs for the filler model')
parser.add_argument('--filler_generated_size', type=int, default=10000, help='define the generated size of the filler model')
parser.add_argument('--decoration_max_len', type=int, default=20, help='define the maximum length for the generated decorations')
parser.add_argument('--batch_size', type=int, default=64, help='define the batch size for the filler and discriminator')
parser.add_argument('--filler_num_decoder_layers', type=int, default=4, help='define the number of transformer decoder layers of the filler model')
parser.add_argument('--filler_dim_feedforward', type=int, default=100, help='define the dimension of the feedforward layer of the filler model')
parser.add_argument('--filler_d_model', type=int, default=128, help='define the dimension of the embedding of the filler model')
parser.add_argument('--filler_num_heads', type=int, default=4, help='define the number of heads of the filler model')
parser.add_argument('--filler_max_lr', type=float, default=1e-5, help='define the maximum learning rate for the filler model')
parser.add_argument('--filler_dropout', type=float, default=0.1, help='define the dropout probability for the filler model')
parser.add_argument('--filler_optimizer', type=str, default='RMSprop', help='select the optimizer (RMSprop or Adam) for training the filler model')

# ===========================
# Discriminator Model
parser.add_argument('--dis_pretrain', action='store_true', help='whether pretraining the discriminator model')
parser.add_argument('--dis_wgan', action='store_true', help='whether applying the WGAN')
parser.add_argument('--dis_minibatch', action='store_true', help='whether applying the mini-batch discrimination')
parser.add_argument('--dis_pre_epochs', type=int, default=10, help='define the pretraining epochs for the discriminator model')
parser.add_argument('--dis_max_lr', type=float, default=1e-5, help='define the maximum learning rate for the discriminator model')
parser.add_argument('--dis_num_encoder_layers', type=int, default=4, help='define the number of transformer encoder layers of the discriminator model')
parser.add_argument('--dis_dim_feedforward', type=int, default=200, help='define the dimension of the feedforward layer of the discriminator model')
parser.add_argument('--dis_d_model', type=int, default=128, help='define the dimension of the embedding of the discriminator model')
parser.add_argument('--dis_num_heads', type=int, default=4, help='define the number of heads of the discriminator model')
parser.add_argument('--dis_dropout', type=float, default=0.1, help='define the dropout probability for the discriminator model')

# ===========================
# Adversarial training
parser.add_argument('--adversarial_train', action='store_true', help='whether adversarial trian the GAN or WGAN')
parser.add_argument('--adv_epochs', type=int, default=100, help='define the training epochs for the GAN or WGAN')
parser.add_argument('--dis_lambda', type=float, default=0.5, help='define the tradeoff between RL and GAN')
parser.add_argument('--adv_lr', type=float, default=2e-5, help='define the learning rate for the GAN or WGAN')
parser.add_argument('--whole_smiles', action='store_false', help='whether using the complete SMILES strings')
parser.add_argument('--properties', type=str, default='druglikeness', help='define the chemical property for molecular generation (druglikeness, solubility, or synthesizability)')
parser.add_argument('--roll_num', type=int, default=8, help='define the rollout times for Monte Carlo search')
parser.add_argument('--update_rate', type=float, default=0.8, help='define the update rate')
parser.add_argument('--save_name', type=int, default=80, help='the name of the loaded model')
args = parser.parse_known_args()[0]

# ===========================
# Model paths
ORIGINAL_DATASET = 'datasets/' + args.dataset_name + '.csv'
PREPROCESSED_FILE = 'datasets/preprocessed_' + args.dataset_name + '_LEN_' + str(args.dec_min_len) + '.csv'
POSITIVE_FILE = 'datasets/' + args.dataset_name + '_LEN_' + str(args.dec_min_len) + '.csv'
NEGATIVE_FILE = 'results/generated_smiles_' + args.dataset_name + '_EPOCH_' + str(args.filler_epochs) + '.csv'
FILLER_PRETRAINED_MODEL = 'results/save_models/pretrained_filler_' + args.dataset_name + '_EPOCH_' + str(args.filler_epochs) + '.pkl'
DIS_PRETRAINED_MODEL = 'results/save_models/pretrained_discriminator_' + args.dataset_name + '_EPOCH_' + str(args.dis_pre_epochs) + '.pkl'
LEFT_PROPERTIES = ['druglikeness', 'solubility', 'synthesizability']
if args.properties in LEFT_PROPERTIES:
    LEFT_PROPERTIES.remove(args.properties)
parser.add_argument('--original_dataset', type=str, default=ORIGINAL_DATASET, help='the path of the original dataset')
parser.add_argument('--preprocessed_file', type=str, default=PREPROCESSED_FILE, help='the path of the preprocessed file')
parser.add_argument('--positive_file', type=str, default=POSITIVE_FILE, help='the path of the positive file')
parser.add_argument('--negative_file', type=str, default=NEGATIVE_FILE, help='the path of the negative file')
parser.add_argument('--filler_pretrained_model', type=str, default=FILLER_PRETRAINED_MODEL, help='the path of the pretrained filler model')
parser.add_argument('--dis_pretrained_model', type=str, default=DIS_PRETRAINED_MODEL, help='the path of the pretrained discriminator model')
parser.add_argument('--left_properties', type=list, default=LEFT_PROPERTIES, help='the names of the left properties')

# ===========================
# Other model paths
if args.dis_lambda == 0:
    MODEL_NAME = 'Naive'
elif args.dis_lambda == 1:
    MODEL_NAME = 'SeqGAN'
else:
    MODEL_NAME = 'ScaffGAN_' + str(args.dis_lambda)

PATHS = 'results/save_models/' + args.dataset_name + '/' + MODEL_NAME + '/rollout_' + str(args.roll_num)  + '/batch_' + str(args.batch_size) + '/' + str(args.properties)
if args.dis_wgan and args.dis_minibatch:
    PATHS = PATHS + '/WGAN'
elif args.dis_lambda and not args.dis_wgan:
    PATHS = PATHS + '/GAN'  

if args.dis_wgan:
    D_STEPS = 1
else:
    D_STEPS = 5

PROPERTY_FILE = PATHS + '/trained_results.csv' 
# Save the generator at each adversarial training epoch
ScaffGAN_FILLER_MODEL = PATHS + '/Epoch_' + str(args.save_name) + '_filler.pkl' 
ScaffGAN_DIS_MODEL = PATHS + '/Epoch_' + str(args.save_name) + '_dis.pkl'
TOP_12_GAN = 'results/' + args.properties + '_EPOCH_' + str(args.adv_epochs) + '_top_12.pdf'
TOP_12_WGAN = 'results/' + args.properties + '_EPOCH_' + str(args.adv_epochs) + '_top_12_w.pdf'
DRAW_DISTRIBUTIONS = 'results/' + args.properties + '.pdf'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='the model name')
parser.add_argument('--paths', type=str, default=PATHS, help='the path of the model')
parser.add_argument('--d_epochs', type=int, default=1, help='train the discriminator D_EPOCHS times every D_STEPS')
parser.add_argument('--d_steps', type=int, default=D_STEPS, help='train the discriminator D_EPOCHS times every D_STEPS')
parser.add_argument('--property_file', type=str, default=PROPERTY_FILE, help='the path for saving the trained properties data')
parser.add_argument('--scaffgan_filler_model', type=str, default=ScaffGAN_FILLER_MODEL, help='the path for saving the trained filler model of the SpotGAN')
parser.add_argument('--scaffgan_dis_model', type=str, default=ScaffGAN_DIS_MODEL, help='the path for saving the trained discriminator model of the SpotGAN')
parser.add_argument('--top_12_gan', type=str, default=TOP_12_GAN, help='the path for saving the top 12 molecular structures of the SpotGAN')
parser.add_argument('--top_12_wgan', type=str, default=TOP_12_WGAN, help='the path for saving the top 12 molecular structures of the SpotWGAN')
parser.add_argument('--draw_distributions', type=str, default=DRAW_DISTRIBUTIONS, help='the path for saving the distributions of the SpotGAN')
parser.add_argument('--gpus', type=int, default=1, help='the name of the CUDA')
parser.add_argument('--device', type=str, default=DEVICE, help='CPU or GPU')

# ===========================
# Test parameters
parser.add_argument('--test', action='store_true', help='whether testing the model')
parser.add_argument('--scaffold_name', type=str, default='A2', help='the name of one of the five scaffolds')
parser.add_argument('--test_size', type=int, default=300, help='the size of the generated test set (larger than 100)')
parser.add_argument('--trained_filler', type=str, default='results/save_models/ZINC_10k/ScaffGAN_0.5/rollout_8/batch_64/DRD2/GAN/Epoch_50_filler.pkl', help='the path to the trained filler model')
parser.add_argument('--test_property', type=str, default='DRD2', help='the desired chemical property to be tested')
args = parser.parse_args()

# ===========================
def main(args):
	# train model
    if not args.test:
        train(args)
    # test model
    elif args.test:
        test(args)
    else:
        print('The command setting is incorrect!')

if __name__ == '__main__':
    main(args)















