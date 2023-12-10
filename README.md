## Partition-and-Debias: Agnostic Biases Mitigation via a Mixture of Biases-Specific Experts [ICCV 2023]
<a href="https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Partition-And-Debias_Agnostic_Biases_Mitigation_via_a_Mixture_of_Biases-Specific_Experts_ICCV_2023_paper.pdf"><img src="https://img.shields.io/static/v1?label=Paper&message=PnD&color=red" height=20.5></a> 
<a href="https://arxiv.org/abs/2308.10005"><img src="https://img.shields.io/badge/arXiv-2308.10005-b31b1b.svg" height=20.5></a>


[**Jiaxuan Li**](https://jiaxuan-li.github.io/)<sup>1</sup>, [**Minh-Duc Vo**](https://vmdlab.github.io/)<sup>1</sup>, [**Hideki Nakayama**](http://www.nlab.ci.i.u-tokyo.ac.jp/index-e.html)<sup>1</sup>

1 The University of Tokyo

> Bias mitigation in image classification has been widely researched, and existing methods have yielded notable results. However, most of these methods implicitly assume that a given image contains only one type of known or unknown bias, failing to consider the complexities of real-world biases. We introduce a more challenging scenario, agnostic biases mitigation, aiming at bias removal regardless of whether the type of bias or the number of types is unknown in the datasets. To address this difficult task, we present the Partition-and-Debias (PnD) method that uses a mixture of biases-specific experts to implicitly divide the bias space into multiple subspaces and a gating module to find a consensus among experts to achieve debiased classification. Experiments on both public and constructed benchmarks demonstrated the efficacy of the PnD.

<div align=center><img width="95%" src="https://github.com/Jiaxuan-Li/PnD/files/12773893/fig_model.pdf"/></div>

## Setup
Install the required packages using conda with the provided [environment.yml](environment.yml) file.

## Training and evaluation

```shell
python main.py \
--dataset bmnist --percent=5.00pct \
--device=cuda:0 \
--exp=PnD_bmnist \
--weight_decay=1e-5 \
--lr1=0.001 --lr2=0.0005 --lr_decay_step1=20 --lr_decay_step2=20 --lr_gamma=0.5 \
--temperature=0.1 --alpha1=0.2 --alpha2=2.0 --beta=4.0 --num_epochs1=70 --num_epochs2=100
```

## Acknowledgements
This repo is built on [OccamNet](https://github.com/erobic/occam-nets-v1) and [DFA](https://github.com/kakaoenterprise/Learning-Debiased-Disentangled), we thank the authors for their great effort.

## Citation
If you find our work helpful for your research, please kindly consider citing:

    @InProceedings{li2023pnd,
        author    = {Li, Jiaxuan and Vo, Duc Minh and Nakayama, Hideki},
        title     = {Partition-And-Debias: Agnostic Biases Mitigation via a Mixture of Biases-Specific Experts},
        booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
        month     = {October},
        year      = {2023},
        pages     = {4924-4934}
    }
