# NConv-CNN
This is the PyTorch implementation for our work:

1. [Propagating Confidences through CNNs for Sparse Data Regression](https://arxiv.org/abs/1805.11913)

2. [Confidence Propagation through CNNs for Guided Sparse Depth Regression ](https://arxiv.org/abs/1811.01791)


If you use this code or compare against it, please cite our work:
```
@article{eldesokey2018propagating,
  title={Propagating Confidences through CNNs for Sparse Data Regression},
  author={Eldesokey, Abdelrahman and Felsberg, Michael and Khan, Fahad Shahbaz},
  journal={arXiv preprint arXiv:1805.11913},
  year={2018}
}
```
```
@article{eldesokey2018confidence,
  title={Confidence Propagation through CNNs for Guided Sparse Depth Regression},
  author={Eldesokey, Abdelrahman and Felsberg, Michael and Khan, Fahad Shahbaz},
  journal={arXiv preprint arXiv:1811.01791},
  year={2018}
}
```
## Contents
0. [Dependencies](#dependencies)
0. [Networks Description](#networks-description)
0. [Evaluation using Pretrained Weights](#evaluation-using-pretrained-weights)
0. [The NYU-Depth-v2 dataset](#the-nyu-depth-v2-dataset)
0. [Contact](#contact)

## Dependecies
* opencv (To save output images)
* json (To read experiment parameters file)

## Usage
```
python run-nconv-cnn.py -mode [MODE] -exp [EXP] -chkpt [CHKPT] -set [SET]
```
`[MODE]:` The mode could be either `train` or `eval`.

`[EXP]:` The name of the directory in 'workspace' which has the network file. 

`[CHKPT]: (optional)`  Continue traing from a specific epoch _or_ evaluate using a specific epoch.

`[SET]: (optional)` The set to evaluate on. The possible options are `val`, `selval` or `test`.

## Networks Description
Networks are located in "workspace" directory. Each network file is stored in its own directory and associated with `params.json` which has the training parameters for the network.

Four netwokrs are available:
1. `exp_unguided_disparity:` Unguided depth completion network trained on disparity *(Deonted as NConv-HMS in paper [1])*
2. `exp_unguided_depth:` Unguided depth completion network trained on depth.
3. `exp_guided_nconv_cnn_l1:` Guided depth completion network trained on depth *(Denoted as MS-Net[LF] or NConv-CNN-L1 in paper [2])*
4. `exp_guided_nconv_cnn_l2:` Guided depth completion network trained on depth *(Denoted as NConv-CNN-L2 in paper [2])*
5. `exp_guided_enc_dec:` Guided depth completion network trained on depth *(Denoted as Enc-Dec[EF] in paper [2])*

## Evaluation using Pretrained Weights 
You can evaluate any of the networks using the pretrained-weights by calling 
```
python run-nconv-cnn.py -mode eval -exp <exp_name>
```

## The NYU-Depth-v2 dataset
The implemntation for the NYU-Depth-v2 dataset can be found at:
https://github.com/abdo-eldesokey/nconv-nyu

## Contact
Abdelrahman Eldesokey

E-mail: abdelrahman.eldesokey@liu.se
