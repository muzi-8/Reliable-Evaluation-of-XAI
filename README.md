# Reliable-Evaluation-of-XAI
This repository is the official implementation and is a benchmark for comprehensive evaluation of model saliency/attention map explanations from  existing interpretability technique.
This repository contains the code, pre-trained model, constructed evaluation data set and ground truth by human annotation. We propose a unified evaluation framework for the 
three dimensions of accuracy, persuasibility and class-discriminativeness.
# Evaluation framework
![image](https://github.com/muzi-8/Reliable-Evaluation-of-XAI/blob/main/images/framework.PNG)
# Contents
## code
## pre-trained model
## Interpretability technique
We evaluated three existing visual interpretation methods via empirical study: Grad-CAM[1],Mask[2],Rise[3].
## constructed evaluation dataset
![image](https://github.com/muzi-8/Reliable-Evaluation-of-XAI/blob/main/images/dataset.PNG)
## Ground Truth
## Evaluation Process
- Overview of Accuracy Evaluation
![image](https://github.com/muzi-8/Reliable-Evaluation-of-XAI/blob/main/images/accuracy%20pipeline.PNG)
- Overview of  Persuasibility Evaluation
![iamge](https://github.com/muzi-8/Reliable-Evaluation-of-XAI/blob/main/images/persuasibility%20pipeline.PNG)
- Overview of Class-discriminativeness Evaluation
# Reference
## paper
1. Selvaraju R R, Cogswell M, Das A, et al. Grad-cam: Visual explanations from deep networks via gradient-based localization[C]//Proceedings of the IEEE international conference on computer vision. 2017: 618-626.
2. Fong R C, Vedaldi A. Interpretable explanations of black boxes by meaningful perturbation[C]//Proceedings of the IEEE International Conference on Computer Vision. 2017: 3429-3437.
3. Petsiuk V, Das A, Saenko K. Rise: Randomized input sampling for explanation of black-box models[J]. arXiv preprint arXiv:1806.07421, 2018.
## code
1. [Grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
2. [Mask](https://github.com/ruthcfong/perturb_explanations)
3. [Rise](https://github.com/eclique/RISE)

# Citation
