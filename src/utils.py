import constants

def get_head_number(layer, head_index):
    """
    Given a layer and a head index, calculate the global head number.
    
    Args:
    layer (int): Zero-indexed number of the layer.
    head_index (int): Zero-indexed number of the head within the given layer.

    Returns:
    int: One-indexed global head number.
    """
    return layer * constants.NUM_HEADS_PER_LAYER + head_index + 1

def get_layer_and_head(head_number):
    """
    Given a one-indexed global head number, calculate the corresponding zero-indexed layer and head index.
    
    Args:
    head_number (int): One-indexed global head number.

    Returns:
    tuple: (layer, head_index), where both are zero-indexed.
    """
    # print("Number of heads per layer: ", constants.NUM_HEADS_PER_LAYER, "Number heads: ", head_number//constants.NUM_HEADS_PER_LAYER, " Head Number: ", head_number)
    head_number -= 1  # Adjust for zero-indexing
    return head_number // constants.NUM_HEADS_PER_LAYER, head_number % constants.NUM_HEADS_PER_LAYER