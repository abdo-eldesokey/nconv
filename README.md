# NConv-CNN
This is the PyTorch implementation for our work:

[Propagating Confidences through CNNs for Sparse Data Regression](https://arxiv.org/abs/1805.11913)

[Confidence Propagation through CNNs for Guided Sparse Depth Regression ](https://arxiv.org/abs/1811.01791)


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
## Dependecies
* opencv (To save output images)
* json (To read experiment parameters file)

## Usage
```
python run-nconv-cnn.py -mode [MODE] -exp [EXP] -chkpt [CHKPT] -set [SET]
```
`[MODE]:` The mode could be either `train` or `eval`.

`[EXP]:` The name of the directory in 'workspace' which has the network file. 

`[CHKPT]:` Continue traing from a specific epoch _or_ evaluate using a specific epoch.

`[SET]:` The set to evaluate on. The possible options are `val`, `selval` or `test`.

## Contact
Abdelrahman Eldesokey

E-mail: abdelrahman.eldesokey@liu.se
