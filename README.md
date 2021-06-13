# Readme

This project contains simple implementations of Image2StyleGAN and Image2StyleGAN++.

**Please note that this is not official implementations and this project is used for a course project.**

We first do some exploratory experiments of Image2StyleGAN: we investigate the optimization in latent space and W space.

Then we reproduce some experiments of Image2StyleGan:

* image reconstruction
* morphing
* style transfer

At last we implement a simple Image2StyleGAN++ model, which contains noise optimization and Three blocks: Masked W+ Optimization, Masked Noise Optimization, Masked Style Transfer.

Pretrained StyleGAN model can be downloaded [here](https://github.com/lernapparat/lernapparat/releases/download/v2019-02-01/karras2019stylegan-ffhq-1024x1024.for_g_all.pt).

* Image2StyleGAN running command:
`python execute.py`

* Image2StyleGAN++ running command:
`python execute_v2.py`


## Exploratory experiments

### latent space

We try to reconstruct those four images.

![image](pictures/explore/Z_space/1.png)

If we use space Z and we get the results as:

![image](pictures/explore/Z_space/2.png)

This show that space Z fails to represent original pictures.

### W space

We use W space to reconstruct the same four pictures, and we get the results shown as:

![image](pictures/explore/W_space/reconstruction/1.png)

We can find that for human face pictures we can reconstruct properly but for other pictures it fails to reconstruct them.

Also we can do morphing in W space:

![image](pictures/explore/W_space/morphing/1.png)



## Experiment of Image2StyleGAN

### Image reconstruction

We try to reconstruct the same pictures in W+ space and we get:

![image](pictures/image2stylegan/reconstruct/1.png)

we can find that not only human face pictures can be reconstructed properly but also other kinds of pictures can be reconstructed.



### Morphing

We do some experiments on image morphing:

![image](pictures/image2stylegan/morphing/1.png)

![2](pictures/image2stylegan/morphing/2.png)

We find that it basically can work but some babbles may appear.



### Style transfer

![image](pictures/image2stylegan/style_transfer/1.png)



## Experiments of Image2StyleGAN++

### Crossover

![image](pictures/image2stylegan++/crossover/1.png)

It basically can work but not very well.

