# Unsupervised Motion Adaptation via DC3

## Introduction
This project aims to propose a new method for motion editing and retargeting that involves close interactions between characters while preserving the spatial relationship to keep the scene semantics unchanged. We plan to reproduce the traditional optimization method where the motions are represented by a simple structure called InteractionMesh as a starting point. As it obtains the optimal results by iterations, which is slow and nondeterministic convergent, we will improve it by integrating it into an unsupervised Deep Neural Network (DNN) under constraints, which we use DC3 at the moment. This method demonstrates its advantages in generalizability, time-efficient and robust performance even when trained on relatively small amounts of data.

## DC3
This project is adapted from DC3, but it does not need the data of simple problem or other problems defined in the original DC3 paper. For more information, visit https://github.com/locuslab/DC3/.

## Instruction
### Running experiments
Run `python method.py` in the root directory to train the model. Arguments can be changed in `default_args.py`

Run `python predict.py` in the root directory to predict the deformed interaction using trained model

## Generate data
Run `python make_datasets.py` in the `datasets` folder

## Visualisation
Go to Blender and import `visualisation.blender` or `visualisation_multiframes.blender` in `data_blender` (which is used to visualise the interaction of multiple frames), then run the script. This will visualise the perdicted motion which is named in particular.

