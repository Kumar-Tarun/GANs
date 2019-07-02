The script *infogan.py* is an implementation of the following paper : *Chen, Xi & Duan, Yan & Houthooft, Rein & Schulman, John & Sutskever, Ilya & Abbeel, Pieter. (2016). InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets.*

Majority of the work is inspired by **https://github.com/eriklindernoren/Keras-GAN/tree/master/infogan.**

However, I did some changes:
For MNIST:
1. Made the network resemble more like the one specified in the paper.
2. Changed the mutual-information loss function, to ignore the *H(c)* part, since it can be assumed to be a constant. 
3. Hyperparameter **λ** is set to 3, to emphasize more on mutual information, which in turn results in a better disentangled representation.
4. Sample the Gaussian noise from **[-1,1]**, instead of original **[0,1]**, since real images are normalized to **[-1,1]**.

For CelebA:
1. Same as above. Changed the dimension of latent code to 250.
2. Same as above.
3. **λ** is set to 2.
4. Sample the Gaussian noise from **[0,1]**.

Some notes:
1. As mentioned in the DCGAN paper, transposed convolutions tend to work better and  help in better stability of GANs, but in my case simply upsampling followed by a vanilla 
convolution layer worked well and produced better results than transposed convolutions.
2. The current implementation covers only discrete latent codes. I plan to add functionality for continuous codes later.