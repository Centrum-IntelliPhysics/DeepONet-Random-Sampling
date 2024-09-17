# Efficient_DeepONet_training
This repository replicates the result of the paper "Efficient Training of Deep Neural Operator Networks via Randomized Sampling."

## **Efficient Training of Deep Neural Operator Networks via Randomized Sampling**
[Sharmila Karumuri](https://scholar.google.com/citations?user=uY1G-S0AAAAJ&hl=en), [Lori Graham-Brady](https://scholar.google.com/citations?user=xhj8q8cAAAAJ&hl=en) and [Somdatta Goswami](https://scholar.google.com/citations?user=GaKrpSkAAAAJ&hl=en).

In this work, we propose a novel random sampling technique during the training of DeepONet aimed at enhancing the generalization ability of the model and substantially reducing computational demands. This technique specifically targets the trunk network of DeepONet, which generates basis functions corresponding to the spatiotemporal locations within a bounded domain where the physical system operates.

Traditionally, DeepONet training involves evaluating the trunk network across a uniform grid of spatiotemporal points to construct the loss function for each iteration. This conventional method, however, results in larger batch sizes, leading to challenges such as poor generalization, slower convergence, and escalated memory demands due to the constraints of the stochastic gradient descent (SGD) optimizer. By adopting random sampling for the inputs of the trunk network, our approach effectively counters these issues, promoting improved generalization and reducing memory usage during training, which translates into significant computational efficiency.

We validate our hypothesis with three benchmark examples that demonstrate notable reductions in training time while maintaining comparable or even better generalization performance than the traditional training approach. The results affirm that integrating randomization into the trunk network inputs during training significantly enhances the efficiency and robustness of DeepONet, marking a promising direction for refining the model’s performance in simulating complex physical systems.

## Installing

The code for examples is written in pytorch. Install dependencies at [requirements.txt](https://github.com/Centrum-IntelliPhysics/Efficient_DeepONet_training/tree/main/requirements.txt) and clone our repository
```
git clone https://Centrum-IntelliPhysics/Efficient_DeepONet_training.git
cd Efficient_DeepONet_training
```
## Repository Overview

This repository contains implementations and analyses for the experiments described in the paper. The repository is organized as follows:

*	Example Folders: Each example discussed in the paper is located in its respective folder. Within these folders, you will find a Python file named DeepONet_analysis.py. This script demonstrates the process of random subsampling of the inputs to the trunk network for efficient training of the DeepONet model.
* Results: Results from the computations are saved in the analysis_results folder. This includes outputs generated from running the DeepONet_analysis.py scripts.
* Postprocessing: The 'postprocessing' folder contains code for generating plots and visualizations based on the analysis results.
  
### Citation:
If you use this code for your research, please cite our paper.
