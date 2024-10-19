## **Efficient Training of Deep Neural Operator Networks via Randomized Sampling**
[Sharmila Karumuri](https://scholar.google.com/citations?user=uY1G-S0AAAAJ&hl=en), [Lori Graham-Brady](https://scholar.google.com/citations?user=xhj8q8cAAAAJ&hl=en) and [Somdatta Goswami](https://scholar.google.com/citations?user=GaKrpSkAAAAJ&hl=en).

“Here are both the presentation slides and the recording of our presentation, which discuss our approach and the results we achieved using it: 
Slides: [DeepONet-randomization-study-final.pptx](https://github.com/user-attachments/files/17442591/DeepONet-randomization-study-final.pptx)
Presentation: https://github.com/user-attachments/assets/90aea455-5b07-446e-b5f0-0d7cf8139f90

In this work, we introduce a novel random sampling technique for training DeepONet, designed to enhance model generalization and reduce computational demands. This technique focuses on the trunk network of DeepONet, which generates basis functions for spatiotemporal locations within a bounded domain where the physical system operates.

Traditionally, DeepONet training involves evaluating the trunk network on a uniform grid of spatiotemporal points to construct the loss function for each iteration. This approach results in larger batch sizes, which can lead to poor generalization, slower convergence, and increased memory usage. Our method, which employs random sampling for the trunk network inputs, addresses these issues by reducing batch sizes, thereby improving generalization and reducing memory usage, ultimately enhancing computational efficiency.

We validate our approach with three benchmark examples, demonstrating significant reductions in training time while maintaining or even improving generalization performance compared to the traditional training approach.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e008fb78-ca0b-419a-b44d-5ce5daaf460e" alt="Traditional Approach" width="600"/>
  <br/>
  <strong>Figure 1: Traditional Approach</strong>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/7d6fa128-cedb-4949-943d-f1d4be83c275" alt="Our Approach" width="600"/>
  <br/>
  <strong>Figure 2: Our Approach</strong>
</p>

## Results
https://github.com/user-attachments/assets/85a38267-98cd-4ce6-ba3a-c304b151dd8f

The dotted lines in the train and test plots represent the traditional training approach, while the solid lines depict our randomized sampling approach. This animation clearly shows that we achieve the same level of test accuracy in one-fifth of the training time.


## Data and Analysis

The labeled dataset used for the problems demonstrated in the manuscript, the data generation script along with the postprocessing results are uploaded [here](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/sgoswam4_jh_edu/EggFTAkvzdhMirX6A3ZglUABetWx7U0l-hfyAuxPuvDWFw?e=PObtYk).

## Installing

The code for examples is written in PyTorch. To install the dependencies, refer to [requirements.txt](https://github.com/Centrum-IntelliPhysics/Efficient_DeepONet_training/tree/main/requirements.txt) and clone our repository:
```
git clone https://Centrum-IntelliPhysics/Efficient_DeepONet_training.git
cd Efficient_DeepONet_training
```
## Repository Overview

This repository contains implementations and analyses for the experiments described in the paper. It is organized as follows:

* Example Folders: Each example discussed in the paper is located in its respective folder. Within these folders, you will find a Python file named DeepONet_analysis.py, which demonstrates the process of random subsampling of the inputs to the trunk network for efficient training of the DeepONet model.
* Results: Results from the computations are saved in the 'analysis_results' folder. This includes outputs generated from running the DeepONet_analysis.py scripts.
* Postprocessing: The 'postprocessing' folder contains code for generating plots and visualizations based on the analysis results.
  
### Citation:
If you use this code for your research, please cite our paper http://arxiv.org/abs/2409.13280.
