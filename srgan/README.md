The scripts *srgan_v\*.py* is an implementation of the following paper : *Ledig, Christian, et al. "Photo-realistic single image super-resolution using a generative adversarial network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.*

Majority of the work is inspired by **https://github.com/eriklindernoren/Keras-GAN/tree/master/srgan.**

However, I did some changes:

Both for v1 and v2:
1. Made the network resemble more like the one specified in the paper.
2. Changed the activation functions in generator to **PReLU**, as specified in the paper.
3. Changed the configuration of Adam optimizer to match that of the paper.
4. Changed the loss function of discriminator to *binary_crossentropy*.
5. VGG feature maps are scaled by *12.75* as mentioned in the paper. 

Some notes:
1. This implementation scales 32x32 images to 128x128 images. 
2. The results in celeba-images-v1/ are decent enough given that the SRGAN generator was not initialized with SRResNet weights.
3. The script for v2 is a little improvement over the v1 implementation. To utilize lower level information better, I first trained the network with VGG loss, that is based on initial layers (last conv layer of 2nd block in VGG19). Then, I intialized the newtork with these weights and trained the network again with both the original VGG loss and new VGG loss in the ratio of **70:30**. This produced better results than v1 and in lesser number of epochs. The adversarial loss proportion was kept as such during both the trainings.