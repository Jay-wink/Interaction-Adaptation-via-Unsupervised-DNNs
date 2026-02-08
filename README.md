# Unsupervised Motion Adaptation via DC3

This project aims to propose a new method for motion editing and retargeting that involves close interactions between characters while preserving the spatial relationship to keep the scene semantics unchanged. We plan to reproduce the traditional optimization method where the motions are represented by a simple structure called InteractionMesh as a starting point. As it obtains the optimal results by iterations, which is slow and nondeterministic convergent, we will improve it by integrating it into an unsupervised Deep Neural Network (DNN) under constraints, which we use DC3 at the moment. This method demonstrates its advantages in generalizability, time-efficient and robust performance even when trained on relatively small amounts of data.

# DC3

For more information, visit https://github.com/locuslab/DC3/.

