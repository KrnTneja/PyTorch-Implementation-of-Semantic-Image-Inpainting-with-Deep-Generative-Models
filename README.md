# PyTorch-Implementation-of-Semantic-Image-Inpainting-with-Deep-Generative-Models

PyTorch Implementation of <a href="http://openaccess.thecvf.com/content_cvpr_2017/papers/Yeh_Semantic_Image_Inpainting_CVPR_2017_paper.pdf">Semantic Image Inpainting with Deep Generative Models</a>.


### Details
1. Latent space dimension: 100
2. GAN model: DCGAN
3. Image size: 64x64x3 (or 96x96x1)
4. Optimizer: ADAM 
5. lambda: 0.003 (Prior weight)
6. Random horizontal flipping: not used
7. Restrict z to [-1,1]: not used
8. Iterations of backprop to input: 1500
9. Poission Blending: Using gradient descent (3000 steps)

## Required Packages
* `pytorch`
* `scikit-image`
* `numpy`
* CUDA

## Training and Inpainting

To train the DCGAN model, store all the images in `./data/`, go to `./model/` and run:

```
$ python train.py
```

To inpaint the images using backprop to input followed by Poission blending, store all the test images in `./test_images/`, go to `./model/` and run:

```
$ python inpaint.py
```

This will randomly create white patches in the image to corrput them and store `original`, `corrupted`, `output` and `blended` images in `./outputs/`.

## TODO

* CPU support
* Examples
