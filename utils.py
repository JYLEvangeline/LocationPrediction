import numpy as np
import os
import random
import torch

def format_list_to_string(l, sep='\n'):
    ret = []
    for e in l:
        if type(e) == float or type(e) == np.float64:
            ret.append(format_float_to_string(e))
        elif type(e) == list or type(e) == tuple:
            # ret.append(format_list_to_string(e, '\t'))
            ret.append(format_list_to_string(e, ' '))
        else:
            ret.append(str(e))
    return sep.join(ret)

def format_float_to_string(f):
    return str.format('{0:.4f}', f)