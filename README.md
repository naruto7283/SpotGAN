# SpotGAN

A PyTorch implementation of SpotGAN: A Reverse-Transformer GAN Generates Scaffold-Constrained Molecules with Property Optimization.
The paper has been accepted by [ECML-PKDD 2023](https://link.springer.com/chapter/10.1007/978-3-031-43412-9_19). ![Overview of SpotGAN](https://github.com/naruto7283/SpotGAN/blob/main/overview.jpg)

## Installation
First, download the code.  
Then, execute the following command:
```
$ conda env create -n spotgan_env -f env.yml
$ source activate spotgan_env
```
Next, unzip the **DRD2_score.sav.zip** to  **DRD2_score.sav**.


## File Description

  - **datasets:** contains the original datasets and preprocessed datasets. Each dataset contains three columns, separated by ";" into scaffolds, decorations, and SMILES strings.
	  - QM9_10k_LEN_3.csv
	  - ZINC_10k_LEN_10.csv
  - **results:** all generated datasets, saved models, and experimental results are saved in this folder.
	  - **save_models:** all training results, pre-trained and trained filler and discriminator models are saved in this folder.
	  - **test:** all test results are saved in this folder.

## Experimental Reproduction

  - SpotGAN on the QM9 dataset with drug-likeness as the optimized property:
  ``` 
  $ python main.py --filler_pretrain --dis_pretrain --adversarial_train
  ```
  - SpotGAN on the QM9 dataset with solubility as the optimized property:
  ```
  $ python main.py --filler_pretrain --dis_pretrain --adversarial_train --properties 'solubility'
  ```
  - SpotGAN on the QM9 dataset with synthesizability as the optimized property:
  ```  
  $ python main.py --filler_pretrain --dis_pretrain --adversarial_train --properties 'synthesizability'
  ```
  - SpotGAN on the ZINC dataset with drug-likeness as the optimized property:
  ```
  $ python main.py --filler_pretrain --dis_pretrain --adversarial_train --dataset_name 'ZINC_10k' --dec_min_len 10 --filler_epochs 200 --decoration_max_len 50 --filler_d_model 256 --filler_max_lr 1e-4 --filler_optimizer 'Adam' --adv_epochs 50
  ```
  - SpotGAN on the ZINC dataset with solubility as the optimized property:
  ```
  $ python main.py --filler_pretrain --dis_pretrain --adversarial_train --dataset_name 'ZINC_10k' --dec_min_len 10 --filler_epochs 200 --decoration_max_len 50 --filler_d_model 256 --filler_max_lr 1e-4 --filler_optimizer 'Adam' --adv_epochs 50 --properties 'solubility'
  ```	
  - SpotGAN on the ZINC dataset with synthesizability as the optimized property:
  ```
  $ python main.py --filler_pretrain --dis_pretrain --adversarial_train --dataset_name 'ZINC_10k' --dec_min_len 10 --filler_epochs 200 --decoration_max_len 50 --filler_d_model 256 --filler_max_lr 1e-4 --filler_optimizer 'Adam' --adv_epochs 50 --properties 'synthesizability'
  ```
  - SpotWGAN on the QM9 dataset with drug-likeness as the optimized property:
  ```
  $ python main.py --filler_pretrain --dis_pretrain --adversarial_train --dis_wgan --dis_minibatch --dis_max_lr 1e-4
  ```
  - SpotWGAN on the QM9 dataset with solubility as the optimized property:
  ```
  $ python main.py --filler_pretrain --dis_pretrain --adversarial_train --dis_wgan --dis_minibatch --properties 'solubility' --dis_max_lr 1e-4
  ```
  - SpotWGAN on the QM9 dataset with synthesizability as the optimized property:
  ```
  $ python main.py --filler_pretrain --dis_pretrain --adversarial_train --dis_wgan --dis_minibatch --properties 'synthesizability' --dis_max_lr 1e-4
  ```
  - SpotWGAN on the ZINC dataset with drug-likeness as the optimized property:
  ```
  $ python main.py --filler_pretrain --dis_pretrain --adversarial_train --dis_wgan --dis_minibatch --dataset_name 'ZINC_10k' --dec_min_len 10 --filler_epochs 200 --decoration_max_len 50 --filler_optimizer 'Adam' --filler_d_model 256 --filler_max_lr 1e-4 --dis_max_lr 1e-4 --adv_epochs 50
  ```
  - SpotWGAN on the ZINC dataset with solubility as the optimized property:
  ```
  $ python main.py --filler_pretrain --dis_pretrain --adversarial_train --dis_wgan --dis_minibatch --dataset_name 'ZINC_10k' --dec_min_len 10 --filler_epochs 200 --decoration_max_len 50 --filler_optimizer 'Adam' --filler_d_model 256 --filler_max_lr 1e-4 --dis_max_lr 1e-4 --adv_epochs 50 --properties 'solubility'
  ```
  - SpotWGAN on the ZINC dataset with synthesizability as the optimized property:
  ```
  $ python main.py --filler_pretrain --dis_pretrain --adversarial_train --dis_wgan --dis_minibatch --dataset_name 'ZINC_10k' --dec_min_len 10 --filler_epochs 200 --decoration_max_len 50 --filler_optimizer 'Adam' --filler_d_model 256 --filler_max_lr 1e-4 --dis_max_lr 1e-4 --adv_epochs 50 --properties 'synthesizability'
  ```

## Case Studies on Optimization of Bioactivity (BIO)
  
  - Training process on the ZINC dataset using SpotGAN
  ```
  $ python main.py --filler_pretrain --dis_pretrain --adversarial_train --dataset_name 'ZINC_10k' --dec_min_len 10 --filler_epochs 200 --decoration_max_len 50 --filler_d_model 256 --filler_max_lr 1e-4 --filler_optimizer 'Adam' --adv_epochs 50 --properties 'DRD2'
  ```
  - Test process on the ZINC dataset using SpotGAN: we provide five scaffolds and generate novel molecules with high biological activities on them. The scaffold names can be set to A1, A2, B1, B2, B3, C1, C2, D1, D2, D3, D4, E1, and E2. Numbers represent the number of attachment points on the given scaffold. Experimental results can be reproduced using A2, B3, C2, D4, and E2.
  ```
  $ python main.py --test --scaffold_name 'A2'
  $ python main.py --test --scaffold_name 'B3'
  $ python main.py --test --scaffold_name 'C2'
  $ python main.py --test --scaffold_name 'D4'
  $ python main.py --test --scaffold_name 'E2'
  ```
  
## Citation
  ```
  C. Li and Y. Yamanishi (2023). SpotGAN: A reverse-transformer GAN generates scaffold-constrained molecules with property optimization. ECML-PKDD 2023.
  ```
  
  BibTeX format:
  ```
  @inproceedings{li2023spotgan,
  title={SpotGAN: A Reverse-Transformer GAN Generates Scaffold-Constrained Molecules with Property Optimization},
  author={Li, Chen and Yamanishi, Yoshihiro},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={323--338},
  year={2023},
  organization={Springer}
}
  ```
