# ML-Based Top Taggers: Performance, Uncertainty and Impact of Tower & Tracker Data Integration

![Python 3.7](https://img.shields.io/badge/python-3.7-brightgreen.svg?style=plastic&logo=python)
![](https://img.shields.io/badge/pytorch-1.9.1-brightgreen.svg?style=plastic&logo=pytorch)

<b>Rameswar Sahu, Kirtiman Ghosh</b>

https://doi.org/10.48550/arXiv.2309.01568

**Abstract:** Machine learning algorithms have the capacity to discern intricate features directly from raw data. We demonstrated the performance of top taggers built upon three machine learning architectures: a BDT that uses jet-level variables (high-level features, HLF) as input, while a CNN trained on the jet image, and a GNN trained on the particle cloud representation of a jet utilizing the 4-momentum (low-level features, LLF) of the jet constituents as input. We found significant performance enhancement for all three classes of classifiers when trained on combined data from calorimeter towers and tracker detectors. The high resolution of the tracking data not only improved the classifier performance in the high transverse momentum region, but the information about the distribution and composition of charged and neutral constituents of the fat jets and subjets helped identify the quark/gluon origin of sub-jets and hence enhances top tagging efficiency. The LLF-based classifiers, such as CNN and GNN, exhibit significantly better performance when compared to HLF-based classifiers like BDT, especially in the high transverse momentum region. Nevertheless, the LLF-based classifiers trained on constituents' 4-momentum data exhibit substantial dependency on the jet modeling within Monte Carlo generators. The composite classifiers, formed by stacking a BDT on top of a GNN/CNN, not only enhance the performance of LLF-based classifiers but also mitigate the uncertainties stemming from the showering and hadronization model of the event generator. We have conducted a comprehensive study on the influence of the fat jet's reconstruction and labeling procedure on the efficiency of the classifiers. We have shown the variation of the classifier's performance with the transverse momentum of the fat jet.

<figure>
<p align="center"><img src="./eff.pdf" alt="Tagging Efficiency" width="800"/></p>
<figcaption align = "center"> The variation of truth-level identification efficiency with the jet radius in different transverse momentum ranges. </figcaption>
</figure>


<b>The codes used in our paper are originally provided in [[2]](#References). Here we are providing the codes for convinience so that the interested reader can reproduce our results. If you use the codes please cite there paper.</b>

## Requirements
- `Python>=3.7`, `PyTorch>=1.9`, and `CUDA toolkit>=10.2`.
- Use the following command to install required packages.
    - ```pip install -r requirements.txt```


### Data
All necessary data files for training and testing the CNN and GNN networks are provided in [`./CNN_trk/data/`](./CNN_trk/data/) and [`./GNN_trk/data/`](./GNN_trk/data/). For the CNNtrkBDTtow and GNNtrkBDTtow composite models we only provide the test datasets in .npy format in [`./CNNtrk_BDTtow/`](./CNNtrk_BDTtow/) and [`./GNNtrk_BDTtow/`](./GNNtrk_BDTtow/)

### Training

Training of the CNNtrk model (ResNet) :

```sh
OMP_NUM_THREADS=4 torchrun --nproc_per_node=1 model.py --exp_name=300_500 --datadir='./data/300_500' --logdir='./logs/' --num_block 3 3 3 --hidden 16 32 64 --num_workers=4 --lr=0.001 --batch_size=32
```

Training of the GNNtrk model (LorentzNet) on 4 GPUs:

```sh
OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 top_tagging.py --exp_name=300_500 --datadir='./data/300_500' --logdir='./logs/' --batch_size=16
```

One can assign `--exp_name` to identity data in different pT ranges. 
Model with the best validation accuracy will be saved in log directory as `exp_name/best-val-model.pt`.

### Evaluation
To eveluate a pre-trained model use the argument `--test_mode`. The pre-trained models are to be stored in the log file in the folder named `--exp_name`, where exp_name refer to the pT-range of the dataset.
### Pre-trained Model

The Pre-trained models for the different pT bins are provided in the respective log files under the name `exp_name/best-val-model.pt`.

## Citation
If you find this work helpful, please cite our paper:
```

@article{Sahu:2023uwb,
    author = "Sahu, Rameswar and Ghosh, Kirtiman",
    title = "{ML-Based Top Taggers: Performance, Uncertainty and Impact of Tower \& Tracker Data Integration}",
    eprint = "2309.01568",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "9",
    year = "2023"
}
```
The GNN codes used in our paper is originally used in [[2]](#References). If you use the code please cite there paper:

```

@article{gong2022efficient,
  author={Gong, Shiqi and Meng, Qi and Zhang, Jue and Qu, Huilin and Li, Congqiao and Qian, Sitian and Du, Weitao and Ma, Zhi-Ming and Liu, Tie-Yan},
  title={An efficient Lorentz equivariant graph neural network for jet tagging},
  journal={Journal of High Energy Physics},
  year={2022},
  month={Jul},
  day={05},
  volume={2022},
  number={7},
  pages={30},
  issn={1029-8479},
  doi={10.1007/JHEP07(2022)030},
  url={https://doi.org/10.1007/JHEP07(2022)030}
}
```

The CNN model used in our analysis was discussed in the Deep Learning course 2021 at the University of Amsterdam. If you use the model please cite:

```

@misc{lippe2022uvadlc,
   title        = {{UvA Deep Learning Tutorials}},
   author       = {Phillip Lippe},
   year         = "2022",
   url = {https://uvadlc-notebooks.readthedocs.io/en/latest/}
}
```

## References
[1] Rameswar Sahu, Kirtiman Ghosh "ML-Based Top Taggers: Performance, Uncertainty and Impact of Tower & Tracker Data Integration." arXiv:2309.01568 

[2] Gong, Shiqi, et al. "An efficient Lorentz equivariant graph neural network for jet tagging." JHEP07(2022)030
