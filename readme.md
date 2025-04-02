# batch-normalisation
A manual pytorch implementation of batch normalisation.
## Why?
Because I don't think I have a strong enough understanding of what it is.
## What is Batch Normalisation?
Suppose we have a batch of inputs of size $b$. Batch Normalisation performs standardisation along the batch dimension, much like how we standarise input features before training a model. In this way, the model converges quicker to a minima in the loss function. Furthermore, large or very small inputs will be better rescaled to strongly mitigate the exploding/vanishing gradient problem.

Note that there are also learnable weight $`\gamma = [1]_{i=1}^b`$ and bias $`\beta = [0]_{i = 1}^b`$ parameters which aim to slightly process the standardised input further to help the model better fit the label space. The equation below mathematically details the process.
```math
x_{s} = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
```
Where $`\epsilon`$ is a very small but positive constant that helps prevent division by zero. Also note that this formula is used during training mode, but not eval mode. Furthermore, during training mode, we keep track of the running mean and variance to be used during evaluation.
```math
x_{rm} \leftarrow (1 - m) \cdot x_{rm} + m \cdot x_m
```
Where $`m \in [0,1]`$ is the momentum, $`x_{rm}`$ is the running mean, and $`x_m`$ is the new recorded mean.
```math
x_{rv} \leftarrow (1 - m) \cdot x_{rv} + m \cdot x_v
```
Where $`m \in [0,1]`$ is the momentum, $`x_{rv}`$ is the running variance, and $`x_v`$ is the new recorded variance.
### 1D Batch Normalisation
Suppose we have a batch of vectors of size $`b`$. We will denote each vector as $`x^{(i)} \in \mathbb{R}^f`$ for $`i \in \{1,2,\dots,b\}`$, meaning we can denote the 
batch as a whole as $`x = [x^{(1)}, x^{(2)}, \dots, x^{(b)}]`$. Then we have
```math
\mathrm{E}[x]_k = \frac{1}{b}\sum_{i=1}^bx^{(i)}_k
```
and
```math
\mathrm{Var}[x]_k = \frac{1}{b}\sum_{i = 1}^b(x^{(i)}_k - E[x]_k)^2
```
Notice that we do not apply any bias correction in the variance formula. After computing the mean and variance, we can use the equation above to obtain $`x_s`$, which must have the same dimension as $`x`$.
### 2D Batch Normalisation
This method of batch normalisation applies for $`x \in \mathbb{R}^{b\times c\times h \times w}`$ where $`c`$ is the number of channels (e.g: 3 for a RGB image), $`h`$ is the height of each input tensor/image, and $`w`$ is the width of the tensor/image.

Computing the mean and variance works exactly the same way, but instead of taking them over just the batch dimension, we also do it over the height and width dimensions.

## Implementation and Testing Methodology
I first implement my version of batch normalisation, then use unit tests to compare between the official pytorch implementation and my own.

## Results
Our 1D and 2D batchnorm implementations are exactly the same as those of PyTorch, barring some missing features. Our work has succeeded in capturing the main essence of batch normalisation, and the missing features are just some extras that are usually considered in very niche scenarios.

## References (Informal)
1. https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
2. https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
3. https://arxiv.org/abs/1502.03167
