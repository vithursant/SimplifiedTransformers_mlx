import mlx.core as mx


def exists(val):
    """
    Check if the value is not None.

    Args:
        val: The value to check.

    Returns:
        bool: True if value exists (is not None), False otherwise.
    """
    return val is not None


def default(val, d):
    """
    Return the value if it exists, otherwise return a default value.

    Args:
        val: The value to check.
        d: The default value to return if val is None.

    Returns:
        The value if it exists, otherwise the default value.
    """
    return val if exists(val) else d


def l2norm_mx(t, groups=2):
    # Step 1: Reshape the tensor to separate groups
    original_shape = t.shape
    group_dimension = original_shape[-1] // groups
    reshaped_t = t.reshape(*original_shape[:-1], groups, group_dimension)
    
    # Step 2: Apply L2 normalization across the last dimension
    # Calculate the L2 norm for each group
    norms = mx.linalg.norm(reshaped_t, ord=2, axis=-1, keepdims=True)
    # Avoid division by zero
    norms = mx.where(norms == 0, 1, norms)
    # Normalize
    normalized_t = reshaped_t / norms
    
    # Step 3: Reshape back to the original shape, adjusted for groups
    normalized_t = normalized_t.reshape(*original_shape[:-1], groups * group_dimension)
    
    return normalized_t


def look_around_mx(x, backward=1, forward=0, pad_value=-1, dim=2):
    # t is the size of the dimension along which we want to "look around"
    t = x.shape[dim-1]
    
    # Prepare padding configuration
    # We pad only along the specified dimension (dim-1 to adjust for 0-indexing)
    pad_width = [(0, 0)] * x.ndim  # Start with no padding for all dimensions
    pad_width[dim-1] = (backward, forward)  # Set padding for the specific dimension
    
    # Pad the array
    padded_x = mx.pad(x, pad_width=pad_width, constant_values=pad_value)
    
    # Collect slices
    arrays = []
    for ind in range(forward + backward + 1):
        arrays.append(mx.take(padded_x, indices=mx.array(list(range(ind, ind + t))), axis=dim-1))
    
    # Concatenate the slices along the specified dimension
    result = mx.concatenate(arrays, axis=dim)
    
    return result


def max_neg_values(t):
    return t * -1e5