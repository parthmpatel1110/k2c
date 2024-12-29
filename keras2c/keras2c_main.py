"""
keras2c_main.py
This file is part of keras2c
Copyright 2020 Rory Conlin
Licensed under MIT License
https://github.com/f0uriest/keras2c

Converts keras model to C code
"""

# imports
from keras2c.layer2c import Layers2C
from keras2c.weights2c import Weights2C
from keras2c.io_parsing import layer_type, get_all_io_names, get_layer_io_names, \
    get_model_io_names, flatten
from keras2c.check_model import check_model
from keras2c.make_test_suite import make_test_suite
import numpy as np
import subprocess
from tensorflow.keras import models

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"


def model2c(model, function_name, malloc=False, verbose=True):
    """Generates C code for model.

    Writes main function definition to "function_name.c" and a public header
    with declarations to "function_name.h"

    Args:
        model (tf.keras.Model): Model to convert.
        function_name (str): Name of C function.
        malloc (bool): Whether to allocate variables on the stack or heap.
        verbose (bool): Whether to print info to stdout.

    Returns:
        malloc_vars (list): Names of variables loaded at runtime and stored on the heap.
        stateful (bool): Whether the model must maintain state between calls.
    """

    model_inputs, model_outputs = get_model_io_names(model)
    includes = '#include <math.h> \n'
    includes += '#include <string.h> \n'
    includes += '#include "./k2c/k2c_include.h" \n'
    includes += '#include "./k2c/k2c_tensor_include.h" \n\n'

    if verbose:
        print('Gathering Weights')
    stack_vars, malloc_vars, static_vars = Weights2C(
        model, function_name, malloc).write_weights(verbose)
    stateful = len(static_vars) > 0
    layers = Layers2C(model, malloc).write_layers(verbose)

    function_signature = 'void ' + function_name + '('
    function_signature += ', '.join(['k2c_tensor* ' +
                                     in_nm + '_input' for in_nm in model_inputs]) + ', '
    function_signature += ', '.join(['k2c_tensor* ' +
                                     out_nm + '_output' for out_nm in model_outputs])
    if len(malloc_vars.keys()):
        function_signature += ',' + ','.join(['float* ' +
                                              key for key in malloc_vars.keys()])
    function_signature += ')'

    init_sig, init_fun = gen_function_initialize(function_name, malloc_vars)
    term_sig, term_fun = gen_function_terminate(function_name, malloc_vars)
    reset_sig, reset_fun = gen_function_reset(function_name)

    with open(function_name + '.c', 'w+') as source:
        source.write(includes)
        source.write(static_vars + '\n\n')
        source.write(function_signature)
        source.write(' { \n\n')
        source.write(stack_vars)
        source.write(layers)
        source.write('\n } \n\n')
        source.write(init_fun)
        source.write(term_fun)
        if stateful:
            source.write(reset_fun)

    with open(function_name + '.h', 'w+') as header:
        header.write('#pragma once \n')
        header.write('#include "./k2c/k2c_tensor_include.h" \n')
        header.write(function_signature + '; \n')
        header.write(init_sig + '; \n')
        header.write(term_sig + '; \n')
        if stateful:
            header.write(reset_sig + '; \n')
    try:
        subprocess.run(['astyle', '-n', function_name + '.h'])
        subprocess.run(['astyle', '-n', function_name + '.c'])
    except FileNotFoundError:
        print("astyle not found, {} and {} will not be auto-formatted".format(function_name + ".h", function_name + ".c"))

    return malloc_vars.keys(), stateful


def gen_function_reset(function_name):
    """Writes a reset function for stateful models

    Reset function is used to clear internal state of the model

    Args:
        function_name (str): name of main function

    Returns:
       signature (str): delcaration of the reset function
       function (str): definition of the reset function
    """

    reset_sig = 'void ' + function_name + '_reset_states()'

    reset_fun = reset_sig
    reset_fun += ' { \n\n'
    reset_fun += 'memset(&' + function_name + \
                 '_states,0,sizeof(' + function_name + '_states)); \n'
    reset_fun += "} \n\n"
    return reset_sig, reset_fun


def gen_function_initialize(function_name, malloc_vars):
    """Writes an initialize function

    Initialize function is used to load variables into memory and do other start up tasks

    Args:
        function_name (str): name of main function
        malloc_vars (dict): variables to read in

    Returns:
       signature (str): delcaration of the initialization function
       function (str): definition of the initialization function
    """

    init_sig = 'void ' + function_name + '_initialize('
    init_sig += ','.join(['float** ' +
                          key + ' \n' for key in malloc_vars.keys()])
    init_sig += ')'

    init_fun = init_sig
    init_fun += ' { \n\n'
    for key in malloc_vars.keys():
        fname = function_name + key + ".csv"
        np.savetxt(fname, malloc_vars[key], fmt="%.8e", delimiter=',')
        init_fun += '*' + key + " = k2c_read_array(\"" + \
            fname + "\"," + str(malloc_vars[key].size) + "); \n"
    init_fun += "} \n\n"

    return init_sig, init_fun


def gen_function_terminate(function_name, malloc_vars):
    """Writes a terminate function

    Terminate function is used to deallocate memory after completion

    Args:
        function_name (str): name of main function
        malloc_vars (dict): variables to deallocate

    Returns:
       signature (str): delcaration of the terminate function
       function (str): definition of the terminate function
    """

    term_sig = 'void ' + function_name + '_terminate('
    term_sig += ','.join(['float* ' +
                          key for key in malloc_vars.keys()])
    term_sig += ')'

    term_fun = term_sig
    term_fun += ' { \n\n'
    for key in malloc_vars.keys():
        term_fun += "free(" + key + "); \n"
    term_fun += "} \n\n"

    return term_sig, term_fun

def k2c(model, function_name, malloc=False, num_tests=10, verbose=True):
    """Converts keras model to C code and generates test suite.

    Args:
        model (tf.keras.Model or str): Model to convert or path to saved .h5 file.
        function_name (str): Name of main function.
        malloc (bool): Whether to allocate variables on the stack or heap.
        num_tests (int): How many tests to generate in the test suite.
        verbose (bool): Whether to print progress.

    Raises:
        ValueError: If model is not an instance of tf.keras.models.Model.

    Returns:
        None
    """

    function_name = str(function_name)
    if isinstance(model, str):
        model = models.load_model(model, compile=False)
    elif not isinstance(model, models.Model):
        raise ValueError('Unknown model type. Model should '
                         'either be an instance of tf.keras.models.Model, '
                         'or a filepath to a saved .h5 model')

    # Check that the model can be converted
    check_model(model, function_name)
    if verbose:
        print('All checks passed')

    malloc_vars, stateful = model2c(model, function_name, malloc, verbose)

    s = 'Done \n'
    s += f"C code is in '{function_name}.c' with header file '{function_name}.h' \n"
    if num_tests > 0:
        make_test_suite(model, function_name, malloc_vars,
                        num_tests, stateful, verbose)
        s += f"Tests are in '{function_name}_test_suite.c' \n"
    if malloc:
        s += "Weight arrays are in .csv files. Place them in the directory from which the main program is run."
    if verbose:
        print(s)
