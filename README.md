# Few-shot-Learning-with-Multiple-Projection-heads

Transferring features from the encoder (feature extractor) yields task-agnostic representations.
Imprinting from the task-agnostic layer may help the model improve generalization on novels classes.
The model has learned intrinsic properties about the labels.
This work introduces projection heads in-between feature extractors and the linear classification layer.
We aim to encourage the model to map representations from the encoder to other latent spaces, which allow more transformation
to be formed, and abstract features can be learned to improve generalization on novel classes.

We trained on CUB-200-2011 dataset.
We split the training (base class) and novel classes into 100 classes each.
Download the dataset and store it on "Data" folder.

We trained the model on one projection head (single fully-connnected layer), two projection heads two FC layers and three FC layers.
Each file contain codes for the experiment.

Execute the run.sh file
We trained on different latent dimensions.
The results for training is stored in "save_pretrain", and for testing on "save_imprinting".


Parts of the code is borrowd from https://github.com/YU1ut.
