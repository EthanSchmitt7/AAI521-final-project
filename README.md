# AAI521-final-project
# Ethan Schmitt - 11/2024
# With any questions feel free to reach out to eschmitt@sandiego.edu


# Galaxy Zoo Kaggle Competition
Competition Link: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge

# Data
The best method for getting the code working would be to download the competition data and unzip each of the artifacts here: data/unzipped

# Functions
I found that many of my scripts required the same functionalities over and over again so I created a functions.ipynb to automate common workflow tasks and they are used in many of the scripts. For more details on how they work, take a look in that file.

# EDA
For my EDA and augmentation approaches, see the script called EDA.ipynb

# Models
Models were made in pytorch and are saved in the models directory. Each subdirectory has a model.py where the model is defined and a saved directory where the saved versions of the models will be saved off.

# Training
See the training.ipynb script to see how the models were trained. It should be noted that the single-head first class group model is trained in the training_sc_model.ipynb script because it necessitated a different process.

# Hyperparameter Tuning
For insight into how I optimized my hyperparameters you can see how I explored the space in the continued_training.ipynb