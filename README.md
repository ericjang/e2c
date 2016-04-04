# e2c

TensorFlow impementation of: [Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images](http://arxiv.org/abs/1506.07365), with code optimized for clarity and simplicity.

![latent](http://i.imgur.com/zO5G3K0.png)

Only 160 lines of code, and only uses Python modules that come installed with TensorFlow. Proper writeup explaining the paper plus improved model code to soon follow.

## Results

Left column are x_t, x_{t+1}, and right column are the E2C reconstructions.
![reconstruction](https://1.bp.blogspot.com/-L2qTQr8XZMY/Vv3cgLAklqI/AAAAAAAAE8g/rjMk2Z98XxEalKyXvtZUGeHtArdsD2vBg/s640/figure_1.png)

Larger step sizes (magnitude of u) yield better latent space reconstruction...

![unfolding latent space](http://i.imgur.com/DF6Gd96.gif)

but degrade image reconstruction fidelity (more on this later...). Here's a different set of obstacles:

![poor reconstruction](http://i.imgur.com/cl9RjlR.png)

## Features:
- Implements the standard E2C model with the factorized Gaussian KL divergence term (Eq. 14)
- Adam Optimizer + Orthogonal weight initialization scheme by [Saxe et al.](http://arxiv.org/abs/1312.6120).
- Learns the latent space of the planar control task (uses the same parameters described in the paper, Appendix B.6.2)

## Training the Model

First, generate the synthetic training data `plane2.npz` by running the following script.

```bash
$ python plane_data2.py
```

Then, train the model
```bash
$ python e2c.py
```

You can then generate visualizations by executing:

```bash
$ python viz_results.py
```

## Acknowledgements

Thanks to Manuel Watter for answering my questions about the paper.

## License

Apache 2.0
