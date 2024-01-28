<div  align="center">

# fouriax

A port of Christian Steinmetz's [`auraloss`](https://github.com/csteinmetz1/auraloss) for [`jax`](https://github.com/google/jax).

</div>

## Setup

```
pip install fouriax
```

## Usage

```python
import jax
import fouriax.stft as stft
from jax.nn.initializers import lecun_normal

key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key)
shape = (4, 4098, 1)

# Initialize the tensor using LeCun normal distribution
input = lecun_normal()(key1, shape)
target = lecun_normal()(key2, shape)
fft_sizes = [1024, 2048, 512]
hop_sizes = [120, 240, 50]
win_lengths = [600, 1200, 240]
params = [
    stft.init_stft_params(x, y, z)
    for x, y, z in zip(fft_sizes, hop_sizes, win_lengths)
]
loss = multi_resolution_stft_loss(params, input, target)
```


# Loss functions

We categorize the loss functions as either time-domain or frequency-domain approaches.
Additionally, we include perceptual transforms.

<table>
    <tr>
        <th>Loss function</th>
        <th>Interface</th>
        <th>Reference</th>
    </tr>
    <tr>
        <td colspan="3" align="center"><b>Time domain</b></td>
    </tr>
    <tr>
        <td>Error-to-signal ratio (ESR)</td>
        <td><code>fouriax.time.esr_loss()</code></td>
        <td><a href=https://arxiv.org/abs/1911.08922>Wright & Välimäki, 2019</a></td>
    </tr>
    <tr>
        <td>Log hyperbolic cosine (Log-cosh)</td>
        <td><code>fouriax.time.log_cosh_loss()</code></td>
        <td><a href=https://openreview.net/forum?id=rkglvsC9Ym>Chen et al., 2019</a></td>
    </tr>
    <tr>
        <td colspan="3" align="center"><b>Frequency domain</b></td>
    </tr>
    <tr>
        <td>Aggregate STFT</td>
        <td><code>fouriax.freq.stft_loss()</code></td>
        <td><a href=https://arxiv.org/abs/1808.06719>Arik et al., 2018</a></td>
    </tr>
    <tr>
        <td>Multi-resolution STFT</td>
        <td><code>fouriax.freq.multi_resolution_stft_loss()</code></td>
        <td><a href=https://arxiv.org/abs/1910.11480>Yamamoto et al., 2019*</a></td>
    </tr>
    <tr>
        <td colspan="3" align="center"><b>Perceptual transforms</b></td>
    </tr>
    <tr>
        <td>FIR pre-emphasis filters</td>
        <td><code>fouriax.perceptual.fir_filter()</code></td>
        <td><a href=https://arxiv.org/abs/1911.08922>Wright & Välimäki, 2019</a></td>
    </tr>
</table>

\* [Wang et al., 2019](https://arxiv.org/abs/1904.12088) also propose a multi-resolution spectral loss (that [Engel et al., 2020](https://arxiv.org/abs/2001.04643) follow),
but they do not include both the log magnitude (L1 distance) and spectral convergence terms, introduced in [Arik et al., 2018](https://arxiv.org/abs/1808.0671), and then extended for the multi-resolution case in [Yamamoto et al., 2019](https://arxiv.org/abs/1910.11480).

# Development

Run tests locally with pytest.

```make test```
