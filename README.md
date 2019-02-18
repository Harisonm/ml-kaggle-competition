# Project Structure

```
4ibd-s1-project-ml
|   README.md                                       <= Read it now
|   .gitignore                                      <= Don't forget using this
|
└─── default                                        <= Folder of Project
|   |
|   └─── notebooks                                  <= Put your notebook on this folder
|   |   |   largerCNN.ipynb                         <= Test of Larger Convolutional Neuronal Network
|   |   |   SampleCNN.ipynb                         <= Test of Sample Convolutional Neuronal Network
|   |   |   MLP.ipynb                               <= Test of Multi Layer Perceptron
|   |   |   SLP.ipynb                               <= Test of Sample perceptron
|   |   |
|   |   └─── tensorboardNotebooks                   <= Tensorboard Folder generate by your notebook
|   |   |   └─── logs                               <= Logs of model Tensorboard                
|   |   |
|   |   └─── testModel                              <= You can create many notebook in this folder to Test model
|   |       └─── Test                               <= Stock your test of this folder 
|   |                                               (this folder will be delete when we doing last commit of project)
|   |
|   └─── src                                        <= Folder of model(mlp,slp,cnn,rnn) type .py
|   |   └─── mModel                                 <= Class of Run Many Models Machine/Deep learning
|           |   main.py
|           |   README.md
|           |   
|           └─── mModel                                 <= Class of Model
|               └─── model                              <= Folder of Model
|                   |   modelManager.py                 <= Class to manage data, preprocessing, tensorboard...
|                   |   cnn.py                          <= Convolutional Neuronal Network Class
|                   |   mlp.py                          <= Multi Layer Perceptron Class
|                   |   slp.py                          <= Sample Layer Class
|                   |   rnn.py                          <= Recurrent Neural Networks Class
|
└─── utils
|   |   README.md                                   <= Readme about Docker
|   |
|   └─── anaconda3                                  <= Anaconda3 Docker
|   |   |   docker-compose.yml
|   |   |   docker.env
|   |   |
|   |   └─── jupyter                                <= Folder of jupyter
|   |       |   Dockerfile
|   |  
|   └─── python3                                    <= Python3 Docker
|       |   docker.env                              <= Environnement variables
|       |   docker-compose.yml                      <= Configuration docker
|       |
|       └─── python
|           |   Dockerfile                          <= Dockerfile settings
|           |   requirements.txt                    <= List of libraries to your python
```

# Run programm
```
python -m default.apps.src.mModel.main "model_name" nbr_epochs
```

# Model Using

After having installed and configured Keras on their machine, students will have to study different models on Datasets by varying the different hyper parameters of these datasets:
* Previous models
  - Linear Model
  - Perceptron Multilayer
  
* New models
  - Conv Net (s)
  - ResNets / HighwayNets - RNN (s)
  
Students will be strongly advised to rely on Tensorboard to visualize compare and retranscribe the performances of these models.
It will be important to show for each dataset, for each model:
The influence of all the hyperparameters of the models (structure, activation functions, ...), as well as the parameters of the learning algorithms (learning rate, momentum, ...)

Reference books (books, articles, magazines, websites ...)
- http://www.deeplearningbook.org/
- https://keras.io/

Computer tools to install
Keras / Tensorflow / Jupyter

| Step                    | Description           | Deadline  |
| ----------------------- |:---------------------:| ---------:|
| Intermediate stage      | Study of the CIFAR-10 dataset https://www.cs.toronto.edu/~kriz/cifar.html Render: Jupyter notebook + pdf / Presentation | Sunday 25/02/2018 23h59 |
| Intermediate stage      | Study of Kaggle datasets Rendering: Jupyter notebook + pdf + Powerpoint presentation     | Wednesday 23/05/2018 23h59 |
| Final render | Study of the dataset 'Mystery' Download the Train Set at: http://www.greenkumquat.com/dataset/ And unzip it. (be careful, more than 60GB decompression) Rendering: Jupyter notebook + pdf + Powerpoint presentation | Monday 18/06/2018 23h59 |

# Package Google Cloud Plateform

## Install SDK GCP in your computer
link SDK to Google Cloud Plateform :
global link : https://cloud.google.com/sdk/?hl=fr
- Mac : [link](https://cloud.google.com/sdk/docs/quickstart-macos?hl=fr)
- Windows : [link](https://cloud.google.com/sdk/docs/quickstart-windows?hl=fr)
- debian/ubuntu : [link](https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu?hl=fr)

## Generate SSH Key

### Terminal in Mac
```
ssh-keygen -t rsa

After you confirm the passphrase, the system generates the key pair.
Your identification has been saved in /Users/user/.ssh/id_rsa.
Your public key has been saved in /Users/user/.ssh/id_rsa.pub.
The key fingerprint is:
ae:89:72:0b:85:da:5a:f4:7c:1f:c2:43:fd:c6:44:38 user@mymac.local
The key's randomart image is:
+--[ RSA 2048]----+
|                 |
|         .       |
|        E .      |
|   .   . o       |
|  o . . S .      |
| + + o . +       |
|. + o = o +      |
| o...o * o       |
|.  oo.o .        |
+-----------------+

pbcopy < ~/.ssh/id_rsa.pub
```

### to add ssh key on agent
```
ssh-add ~/.ssh/id_rsa
```

### Show ssh key
```
ssh-add -L
```



### Terminal in Windows
tutorial [link](http://www.kevinsubileau.fr/informatique/astuces-tutoriels/windows-10-client-serveur-ssh-natif.html)


## Create your Instance GCP

You need create your google account (@gmail) and go to the [link](https://cloud.google.com/) and click in "console"

after that you must create your Google Cloud Plateform account.
Now you need create your VM instance, so your need go to navigator menu (left of browser), and go in compute engine->VM instance.

 Step to Create VM :
1. Clic on create VM
2. clic on marketplace
3. choose "machinelearning" and "free"
4. clic on "AISE TensorFlow NVidia GPU Production"
5. clic on "lautch one compute engine"
6. you can change name of instance and add ram
7. choose "nvidia-tesla-p100" GPU
8. clic on "deploy"

# Git

![git Cheat Sheet](https://github.com/Harisonm/4aibd-s1-project-ml/blob/master/docs/man_GIT.jpg)

# Tensorboard

## Start Tensorboard server

Open a terminal window in your root project directory. Run:
```
tensorboard --logdir path
```

Go to the URL it provides OR on windows:
```
http://localhost:6006/
```

# Mlflow

from 4ibd-s1-project-ml run :
```
mlflow ui
```

mlflow ui web :
```
http://localhost:5000/
```


