import numpy as np

import mlx.core as mx

from simplified_transformers import SimplifiedTransformers


if __name__ == "__main__":
    model = SimplifiedTransformers(
        dim=4096,
        depth=6,
        heads=8,
        num_tokens=20000,
    )

    print(model)

    x = np.random.randint(0, 20000, size=(1, 4096))
    out = model(mx.array(x))
    print(out.shape)