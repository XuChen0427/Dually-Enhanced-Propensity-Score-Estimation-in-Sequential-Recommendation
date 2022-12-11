<<<<<<< HEAD
# Dually Enhanced Propensity Score Estimation in Sequential Recommendation
## Xu Chen, joint Ph.D. student of Renming University of China, GSAI and University of Montreal, RALI
Any question, please mail to xc_chen@ruc.edu.cn or chen.xu@umontreal.ca\
Implementation of Semantic Sentence Matching via Interacting Syntax Graphs in COLING 2022

#The project is developed in the Recbole Platform:
https://github.com/RUCAIBox/RecBole

#The original datasets can be found in https://github.com/RUCAIBox/RecSysDatasets

#This project is published based on original experiments in the CIKM2022 paper. 
For other configs and datasets, DEPS will be developed on the further Recbole version.

#All the hyper parameters in the paper are in ~/recbole/properties/SquenceUnBiasRec/*.yaml



## 1. enter the project dir 
(for the three datasets in the paper, we already process the data to the datasets/. So, if you want to run three datasets in the paper, you do not need to run this one):
 ```bash
python data_process.py
 ```
## 2. run the code for three dataset for 
(for other evaluation style, please ref Recbole doc https://github.com/RUCAIBox/RecBole)
```bash
python run_recbole.py --model=DEPS --dataset=Amazon_Beauty --config_files=recbole/properties/SquenceUnBiasRec/DEPS_amazon_beauty.yaml
```

```bash
python run_recbole.py --model=DEPS --dataset=Amazon_Digital_Music --config_files=recbole/properties/SquenceUnBiasRec/DEPS_amazon_music.yaml
```

```bash
python run_recbole.py --model=DEPS --dataset=mind_small --config_files=recbole/properties/SquenceUnBiasRec/DEPS_mind.yaml
```


##For citation, please cite the following bib
```
@inproceedings{10.1145/3511808.3557299,
author = {Xu, Chen and Xu, Jun and Chen, Xu and Dong, Zhenghua and Wen, Ji-Rong},
title = {Dually Enhanced Propensity Score Estimation in Sequential Recommendation},
year = {2022},
isbn = {9781450392365},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3511808.3557299},
doi = {10.1145/3511808.3557299},
abstract = {Sequential recommender systems train their models based on a large amount of implicit user feedback data and may be subject to biases when users are systematically under/over-exposed to certain items. Unbiased learning based on inverse propensity scores (IPS), which estimate the probability of observing a user-item pair given the historical information, has been proposed to address the issue. In these methods, propensity score estimation is usually limited to the view of item, that is, treating the feedback data as sequences of items that interacted with the users. However, the feedback data can also be treated from the view of user, as the sequences of users that interact with the items. Moreover, the two views can jointly enhance the propensity score estimation. Inspired by the observation, we propose to estimate the propensity scores from the views of user and item, called Dually Enhanced Propensity Score Estimation (DEPS). Specifically, given a target user-item pair and the corresponding item and user interaction sequences, DEPS first constructs a time-aware causal graph to represent the user-item observational probability. According to the graph, two complementary propensity scores are estimated from the views of item and user, respectively, based on the same set of user feedback data. Finally, two transformers are designed to make use of the two propensity scores and make the final preference prediction. Theoretical analysis showed the unbiasedness and variance of DEPS. Experimental results on three publicly available benchmarks and a proprietary industrial dataset demonstrated that DEPS can significantly outperform the state-of-the-art baselines.},
booktitle = {Proceedings of the 31st ACM International Conference on Information & Knowledge Management},
pages = {2260–2269},
numpages = {10},
keywords = {propensity score estimation, sequential recommendation},
location = {Atlanta, GA, USA},
series = {CIKM '22}
}


```
=======
# Dually-Enhanced-Propensity-Score-Estimation-in-Sequential-Recommendation
The implementation of paper Dually Enhanced Propensity Score Estimation in Sequential Recommendation in CIKM2022

Implemented by Chen Xu, Gaoling school of Artificial Intelligence, Renmin University of China, China.
The code implementation will be public after the Conference of CIKM2022.
>>>>>>> d343fc337ce05b6abea54bf066f74fd9cbbb156c
