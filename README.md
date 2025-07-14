Fusion of Modulation Spectrogram and SSL with Multi-head Attention for Fake Speech Detection
===============
<!-- This repository contains our implementation of the paper published in the Speaker Odyssey 2022 workshop, "Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation". This work produced state-of-the-art result on more challenging ASVspoof 2021 LA and DF database.

[Paper link here](https://arxiv.org/abs/2202.12233) -->

This repository contains our implementation of the our paper currently under review for publication at APSIPA ASC 2025. The work focuses on enhancing domain generalizability in fake speech detection, with experiments conducted on the ASVspoof 2019 LA, ASVspoof 2021 LA, and MLAAD datasets. Additionally, the study explores language robustness in multilingual settings.

This repository is built on the baseline system proposed in [SSL-AAISIST](https://github.com/TakHemlata/SSL_Anti-spoofing).

## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ git clone https://github.com/rishithSadashiv/ssl-ms-fsd.git
$ conda create -n ssl_ms_fsd=3.7
$ conda activate ssl_ms_fsd
$ pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
$ cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
(This fairseq folder can also be downloaded from https://github.com/pytorch/fairseq/tree/a54021305d6b3c4c5959ac9395135f63202db8f1)
$ pip install --editable ./
$ cd ..
$ pip install -r requirements.txt
```


## Experiments

### Dataset
Experiments were conducted using the ASVspoof 2019, ASVspoof 2021, and MLAAD datasets. Model training was performed on the ASVspoof 2019 and MLAAD datasets, while evaluation was carried out across all three datasets to assess generalization capabilities.

The datasets used in this work can be accessed from the following sources:
- ASVspoof 2019 LA: [Download here](https://datashare.is.ed.ac.uk/handle/10283/3336).
- ASVspoof 2021 LA: [Download here](https://zenodo.org/record/4837263#.YnDIinYzZhE)
- ASVspoof 2021 LA metadata and labels: [Available here](https://www.asvspoof.org/index2021.html)
- MLAAD dataset (Fake speech only): [Download here](https://deepfake-total.com/mlaad)
- M-AILABS dataset (For bonafide speech in conjunction with MLAAD): [Download here](https://github.com/imdatceleste/m-ailabs-dataset)
- MLAAD protocols: [Available here](doi.org/10.5281/zenodo.11593133)


All protocols used in our work are provided in ``` database/ ``` folder. 

### Pre-trained XLSR model (0.3B)
The XLSR model can be downloaded from [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/xlsr)

### Training on ASVspoof 2019 LA
To train the proposed model on ASVspoof 2019 LA dataset, run:
```
python main_SSL_LA5.py --track=LA --lr=0.000001 --batch_size=14 --loss=WCE --database_path='[Path to dataset]' --protocols_path='[Path to protocols]'
```

### Training on MLAAD 
To train model on MLAAD dataset, run:
- Baseline model:
```
python main_SSL_LA_trainMlaad.py --lr=0.000001 --batch_size=14 --loss=WCE --protocols_path='database/'
```
- Proposed model:
```
python main_SSL_LA5_trainMlaad.py --lr=0.000001 --batch_size=14 --loss=WCE --protocols_path='database/'
```


### Training on Combined dataset
To train model on a combination of ASVspoof 2019 and MLAAD datasets, run:
- Baseline model:
```
python main_SSL_LA_trainCombined.py --lr=0.000001 --batch_size=14 --loss=WCE --protocols_path='database/'
```
- Proposed model:
```
python main_SSL_LA5_trainCombined.py --lr=0.000001 --batch_size=14 --loss=WCE --protocols_path='database/'
```




### Testing trained models

- To evaluate proposed model on ASVspoof 2019 LA dataset, use ```test_asvspoof2019.ipynb``` notebook.
- To evaluate proposed model on MLAAD dataset, use ```test_mlaad.ipynb``` notebook.
- To evaluate proposed model on ASVspoof 2021 LA dataset, use the commands below:
```
python main_SSL_LA5.py --track=LA --is_eval --eval --model_path='[path to model]' --eval_output='eval_scores_LA2021.txt' --database_path='[path to LA 2021 dataset]'
python evaluate_2021_LA.py eval_scores_LA2021.txt ./LA-keys-stage-1/keys/ eval
```

The trained models are provided [here](https://drive.google.com/drive/folders/18dWR2b4ektPid4C8HxFMnECm_loSWF_7?usp=sharing)


## Contact
For queries, please contact:
- Rishith Sadashiv T N: ee24dp010[at]iitdh[dot]ac[dot]in
<!-- ## Citation -->
<!-- If you use this code in your research please use the following citation: -->
<!-- ```bibtex

@inproceedings{tak2022automatic,
  title={Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation},
  author={Tak, Hemlata and Todisco, Massimiliano and Wang, Xin and Jung, Jee-weon and Yamagishi, Junichi and Evans, Nicholas},
  booktitle={The Speaker and Language Recognition Workshop},
  year={2022}
}
``` -->

