# Deep Convolutional Network Based on Anosov and Zadeh

This project should be presented as a new deep convolutional architecture whose local kernels are built from two mathematical ideas:

- Anosov side: stable and unstable directions define anisotropic local geometry.
- Zadeh side: fuzzy membership functions define soft compatibility between the center pixel and its neighbors.

The core layer is `AZConv2d` in [models/azconv.py](/C:/Users/AIRC2/Desktop/murav/models/azconv.py).

Its local rule-wise kernel has the form:

`K_r(x, y) = mu_r(x) * mu_r(y) * kappa_r(x, y)`

where:

- `mu_r(x)` is the Zadeh-style fuzzy membership of point `x` to rule `r`
- `kappa_r(x, y)` is the Anosov-style anisotropic kernel aligned with stable/unstable directions

For the strict thesis model, use variant `az_thesis` from [configs/drive.yaml](/C:/Users/AIRC2/Desktop/murav/configs/drive.yaml).

## What To Say At The Defense

Short formulation:

"I propose a deep convolutional network based on Anosov and Zadeh, where the convolution kernel itself jointly encodes hyperbolic anisotropic geometry and fuzzy membership relations."

Even simpler:

"This is not a post-hoc hybrid of two branches. It is a single convolutional operator in which Anosov geometry determines where and along which directions to aggregate, and Zadeh fuzzy membership determines how strongly neighboring responses should contribute."

## What Exactly Comes From Anosov

- stable/unstable splitting
- anisotropic response along two structurally different directions
- fixed cat-map geometry in the strict current implementation
- hyperbolicity regularization so the geometry does not collapse to isotropic convolution

## What Exactly Comes From Zadeh

- rule-wise membership functions `mu_r(x)` in `[0, 1]`
- fuzzy compatibility between center and neighbor through `mu_r(center) * mu_r(neighbor)`
- interpretable soft assignment instead of hard routing

## Honest Scientific Position

Safe claim:

- "This is a new research architecture motivated by Anosov hyperbolic geometry and Zadeh fuzzy logic."

Safe stronger claim:

- "The novelty is in embedding both structures into the kernel operator itself rather than mixing separate outputs after convolution."

What should not be claimed without stronger benchmarks:

- "global SOTA on DRIVE"
- "formal Anosov dynamical system implemented exactly by the network"

The current implementation is a deep CNN with Anosov-Zadeh kernel operators, not a full dynamical-systems proof object.

## Recommended Naming

Use one of these consistently:

- `AZNet`
- `AZConvNet`
- `Anosov-Zadeh Convolutional Network`
- `Deep Convolutional Network Based on Anosov and Zadeh`

For the thesis text, the cleanest phrasing is:

"Deep convolutional network based on Anosov hyperbolic geometry and Zadeh fuzzy membership."

## Practical Message

If the committee asks "where are Anosov and Zadeh here exactly?", answer:

1. Anosov is in the anisotropic stable/unstable geometry of the kernel.
2. Zadeh is in the fuzzy rule memberships inside the kernel weights.
3. The network is convolutional because aggregation is still local and translation-applied over the image.
4. The novelty is that both structures live inside one operator.
