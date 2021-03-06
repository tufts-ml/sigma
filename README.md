This repo includes the implementation of ICML2021 paper **"Stochastic Iterative Graph Matching (SIGMA)"** (https://arxiv.org/pdf/2106.02206.pdf)

The repo covers the three experiments given in the manuscript.
* Common Graph Matching: an example of matching the PPI network with 5\% noise ratio.
* RDM Pattern Matching in KEGG RPAIR: a demo for finding RDM patterns.
* Image Keypoints Matching: matching keypoints on PASCAL VOC with Berkeley keypoint annotations, built upon DGMC [(DGMC github)](https://github.com/rusty1s/deep-graph-matching-consensus).

# Data
- Common Graph Matching contains BA and PPI datasets. To extract data, we follow instructions from *Scalable Gromov-Wasserstein Learning for Graph Partitioning and Matching*  [(S-GWL github)](https://github.com/HongtengXu/s-gwl).
- RDM Pattern Matching in KEGG RPAIR: we purchased KEGG RPAIR dataset. We do not have license to share this dataset in public domain. 
- Image Keypoints Matching contains PASCAL VOC and SPair71k. PASCAL VOC dataset is extracted following DGMC. [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) is used to download and preprocess the dataset. Download and preprocessing will start when running train.py. For SPair71k dataset, please follow BB-GM and check [(SPair71k link)](https://github.com/martius-lab/blackbox-deep-graph-matching).


# Requirements
The model is tested on a Linux server (32 cores and a A100 GPU) with the following packages,
* pytorch (1.7.1)
* torch-geometric (1.6.3)
* scipy (1.5.4)

# Run examples
We have put each experiment in their own folder.
In each folder, the file `train.py` provides both training and inference procedures. To run an example,
```
$ cd common-graph-matching
$ python train.py
```
