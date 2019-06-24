# IntroVAE

## About this Project
This project implements the IntroVAE.

Please see my project on implementing a [VAE](https://github.com/yukunchen113/VariationalAutoEncoder) for more on VAE latent space analysis.

Model checkpoints, parameters files and run files are saved in the predictions folder. See _Analysis_ section below for more on each run. The model files are called model.ckpt

## Requirements
Please pull my utils repo, and add it to your python path. The functions there are used.

The model training is contained in main.py, the parameters are in params.py and the predictions folder contains saved images during training. The models.py file contains the model for the VAE

## IntroVAE background
IntroVAE is influenced by two parts, a VAE for it's stable latent space, and a GAN for it's high quality image generations. Here, a second objective is added to the VAE, which is for making high quality images. IntroVAE uses the mean squared error loss as a skeleton structure for the GAN, preventing problems like mode collapse. The reconstruction loss term is used as a prior to the model, and the GAN portion will be used to reconstruct better images.

### Resources
Check out:
- [VAES implementation](https://github.com/yukunchen113/VariationalAutoEncoder) and [VAES Paper](https://arxiv.org/abs/1312.6114)
- [EBGANS](https://arxiv.org/abs/1609.03126) is the gan architecture used.
- [IntroVAE original paper](https://arxiv.org/abs/1807.06358) for more information on the model architecture.

### From VAE to IntroVAE
The architecture of the VAE remains unchanged, there are not additional neurons used. The difference, is where the encoder and decoder are switched along the model pipeline. This is to facilitate the use of an EBGAN. The encoder becomes the discriminator, and the decoder becomes the generator. The latent vector doubles as a energy representation vector when the encoder is being used as a discriminator, and a latent vector during the regular VAE pass through.  

### Theory

## Analysis:
I found that using different parameters than the paper works better for me. Tuning the three main latent hyperparameters will have a large effect on the type of images that are generated. \alpha controls the amount of GAN, if this is 0, the model reduces down to a regular VAE. _m_ is used to separate between high and low regions of energy. \beta is used as a term to control the weight of the reconstruction loss. \beta is used to ground the images, too much, and your images will look unrealistic, too little, and the GAN collapses into a weird images which contain certain aspects of detail, such as skin tecture, and hair strands, but fails to maintain the structured look of a realistic face, there is also a chance of posterior collapse. See below:
![beta = 0.05, alpha = 0.25, m = 500](results/low_beta1.jpg)

higher beta is better for this, accordingly, _m_ should be tuned to be higher. This is because higher beta causes higher regularization loss, and the bound _m_ should be enough to accomodate the higher ends of the regularization loss. However, this doesn't provide a good latent traversal, so perhaps increasing the beta value would help.

![beta = 0.75, alpha = 0.25, m =1000](tesults/normal1.jpg)


