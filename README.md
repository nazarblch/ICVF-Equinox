# ICVF (Reinforcement Learning from Passive Data via Latent Intentions) in Equinox

This repository contains reproduction code for the paper [Reinforcement Learning from Passive Data via Latent Intentions](https://arxiv.org/abs/2304.04782).
Based on original codebase [https://github.com/dibyaghosh/icvf_release/tree/main](https://github.com/dibyaghosh/icvf_release/tree/main) with some changes to work in Equinox!

### Examples

To train an ICVF agent on the Antmaze dataset, run:

```
python experiments/antmaze/train_icvf.py --env_name=antmaze-large-diverse-v2
```
