# PyTorch-Implementation-of-Semantic-Image-Inpainting-with-Deep-Generative-Models

PyTorch Implementation of Semantic Image Inpainting with Deep Generative Models (http://openaccess.thecvf.com/content_cvpr_2017/papers/Yeh_Semantic_Image_Inpainting_CVPR_2017_paper.pdf)


Details:
1. 100-d input random vector
2. DC GAN
3. 64x64x3 image
4. ADAM optimizer
5. lambda = 0.003 (weight of prior)
6. random horizontla flipping
7. restrict z to [-1,1]
8. 1500 iterations of backpropagation to the input


TODO:
1. (done) Implement GAN 
2. Prepare datasets
3. (done) Loss function for GAN and training script 
4. (done) Implement backprop to input 
5. (done) Loss function for backprop to input 
6. (done) Random patches dataset
7. (done) Weighted mask
8. Poisson blending



