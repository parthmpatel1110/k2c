"""io_parsing.py
This file is part of keras2c
Copyright 2020 Rory Conlin
Licensed under MIT License
https://github.com/f0uriest/keras2c

Helper functions to get input and output names for each layer etc.
"""

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


def layer_type(layer):
    """Gets the type of a layer

    Args:
        layer (keras Layer): layer you want the type of

    Returns:
        type (str): what kind of layer it is. Eg "Dense", "Conv2D", "SimpleRNN"
    """

    return layer.__class__.__name__


def get_all_io_names(model):
    """Gets names of all  node names in the model

    Args:
        model (keras Model): model to parse

    Returns:
        io (list): names of all the nodes in the model
    """

    # a = [get_layer_io_names(layer) for layer in model.layers]

    a = [get_layer_io_names(model.layers[0])]
    for layer in model.layers:
        __, output = get_layer_io_names(layer)
        a.append(output)
    a = list(set(flatten(a)))

    return a


def get_layer_num_io(layer):
    """Gets the number of inputs and outputs for a layer

    Args:
        layer (keras Layer): layer you want to parse

    Returns:
        num_inputs (int): number of input nodes to the layer
        num_outputs (int): number of output nodes from the layer
    """

    # num_inputs = 0
    # error = False
    # while not error:
    #     try:
            
    #         layer.input(num_inputs)
    #         num_inputs += 1
    #     except :
    #         error = True

    # num_outputs = 0
    # error = False
    # while not error:
    #     try:
    #         layer.get_output_at(num_outputs)
    #         num_outputs += 1
    #     except :
    #         error = True

    try:
        num_inputs = layer.input.shape[1]
    except:
        num_inputs = 0
    try:
        num_outputs = layer.output.shape[1]
    except:
        num_outputs = 0
    return num_inputs, num_outputs


def get_layer_io_names(layer):
    """Gets the names of the inputs and outputs of a layer

    Args:
        layer (keras Layer): layer you want to parse

    Returns:
        inputs (list): names of all the input nodes to the layer
        outputs (list): names of all the output nodes from the layer
    """
    # print("get_layer_io_name")
    # print(layer)
    # print(layer.input)
    # print(layer.output)
    # print(layer.name)
    # print(layer_type(layer))
    # print()
    if(layer_type(layer) == "InputLayer"):
    
        return [], []    
    
    return [layer.input.name], [layer.output.name]


def get_model_io_names(model):
    """Gets names of the input and output nodes of the model

    Args:
        model (keras Model): model to parse

    Returns:
        inputs (list): names of all the input nodes
        outputs (list): names of all the output nodes
    """

    num_inputs = len(model.inputs)
    num_outputs = len(model.outputs)
    inputs = []
    outputs = []
    for i in range(num_inputs):
        nm = model.inputs[i].name.split(':')[0].split('/')[0]
        inputs.append(nm)
    for i in range(num_outputs):
        nm = model.outputs[i].name.split(':')[0].split('/')[0]
        outputs.append(nm)
    return inputs, outputs


def flatten(x):
    """Flattens a nested list or tuple

    Args:
        x (list or tuple): nested list or tuple of lists or tuples to flatten

    Returns:
        x (list): flattened input
    """
    if isinstance(x, list) or isinstance(x, tuple):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
