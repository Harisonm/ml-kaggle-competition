# CNN
Convolutional Neural Networks

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

![git Cheat Sheet](man_git.jpg)




