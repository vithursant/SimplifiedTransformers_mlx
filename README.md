# Apple MLX Port of Simplified Transformer Architecture

Welcome to the official repository for the Apple MLX implementation of the
Simplified Transformer architecture, as described in the insightful paper "A
Simple Design Recipe for Deep Transformers". This repository aims to bring the
efficiencies and innovations of the Simplified Transformer model to the Apple
ecosystem, leveraging the power of Apple's MLX framework for enhanced
performance on MAC's GPUs.

Original Code [here](https://github.com/kyegomez/SimplifiedTransformers/tree/main)

## overview
Transformers have become a cornerstone in the field of deep learning, offering remarkable success across a variety of tasks. However, the complexity of standard Transformer blocks, with their intricate combination of attention mechanisms, MLP sub-blocks, skip connections, and normalization layers, makes them prone to fragility where minor alterations can drastically impact training efficiency and model trainability.

Addressing this challenge, our work simplifies the Transformer architecture without sacrificing performance. By applying principles from signal propagation theory and empirical analysis, we've successfully removed several components traditionally deemed essential, such as skip connections, certain projection parameters, sequential sub-block ordering, and normalization layers. The result? A model that not only matches the standard Transformer's training speed and performance but does so with 15% faster training throughput and a 15% reduction in parameter count.

## install
Create a conda environment using the provided environment configuration file.
```bash
conda env create -f apple_mlx.yaml
```

Activate conda environment.
```bash
conda activate apple_mlx
```

## quick start
```bash
python example.py
```

# acknowledgments
I'd like to extend my gratitude to the authors of the original paper for their work on simplifying Transformer architectures.
