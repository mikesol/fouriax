# fouriax

[![PyPI](https://img.shields.io/pypi/v/fouriax?style=flat-square)](https://pypi.python.org/pypi/fouriax/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fouriax?style=flat-square)](https://pypi.python.org/pypi/fouriax/)
[![PyPI - License](https://img.shields.io/pypi/l/fouriax?style=flat-square)](https://pypi.python.org/pypi/fouriax/)
[![Coookiecutter - Wolt](https://img.shields.io/badge/cookiecutter-Wolt-00c2e8?style=flat-square&logo=cookiecutter&logoColor=D4AA00&link=https://github.com/woltapp/wolt-python-package-cookiecutter)](https://github.com/woltapp/wolt-python-package-cookiecutter)


---

**Documentation**: [https://mikesol.github.io/fouriax](https://mikesol.github.io/fouriax)

**Source Code**: [https://github.com/mikesol/fouriax](https://github.com/mikesol/fouriax)

**PyPI**: [https://pypi.org/project/fouriax/](https://pypi.org/project/fouriax/)

---

A jax port of [`auraloss`](https://github.com/csteinmetz1/auraloss).

## Installation

```sh
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
        <td>DC error (DC)</td>
        <td><code>auraloss.time.DCLoss()</code></td>
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


# PVC

A partial port of core routines in Paul Koonce's [PVC](https://www.cs.princeton.edu/courses/archive/spr99/cs325/koonce.html) can be found in [`pvc.py`](./src/fouriax/pvc.py). This includes a novel FFT algorithm called `fkt` (Fast Koonce Transform) that, combined with `convert`, produces amplitude/frequency pairs for a given signal. This is often more attractive to use in loss functions than a garden-variety FFT because it provides better frequency information.

There is also a `noscbank` method that allows for resynthesis. This can be used as a simple recurrent layer at the end of a network to do waveform synthesis.

## Development

* Clone this repository
* Requirements:
  * [Poetry](https://python-poetry.org/)
  * Python 3.7+
* Create a virtual environment and install the dependencies

```sh
poetry install
```

* Activate the virtual environment

```sh
poetry shell
```

### Testing

```sh
pytest
```

### Documentation

The documentation is automatically generated from the content of the [docs directory](./docs) and from the docstrings
 of the public signatures of the source code. The documentation is updated and published as a [Github project page
 ](https://pages.github.com/) automatically as part each release.

### Releasing

Trigger the [Draft release workflow](https://github.com/mikesol/fouriax/actions/workflows/draft_release.yml)
(press _Run workflow_). This will update the changelog & version and create a GitHub release which is in _Draft_ state.

Find the draft release from the
[GitHub releases](https://github.com/mikesol/fouriax/releases) and publish it. When
 a release is published, it'll trigger [release](https://github.com/mikesol/fouriax/blob/master/.github/workflows/release.yml) workflow which creates PyPI
 release and deploys updated documentation.

### Pre-commit

Pre-commit hooks run all the auto-formatters (e.g. `black`, `isort`), linters (e.g. `mypy`, `flake8`), and other quality
 checks to make sure the changeset is in good shape before a commit/push happens.

You can install the hooks with (runs for each commit):

```sh
pre-commit install
```

Or if you want them to run only for each push:

```sh
pre-commit install -t pre-push
```

Or if you want e.g. want to run all checks manually for all files:

```sh
pre-commit run --all-files
```

---

This project was generated using the [wolt-python-package-cookiecutter](https://github.com/woltapp/wolt-python-package-cookiecutter) template.
