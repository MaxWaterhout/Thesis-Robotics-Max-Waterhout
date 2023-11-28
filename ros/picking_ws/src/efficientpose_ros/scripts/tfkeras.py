
"""
Source Code from Keras EfficientDet implementation (https://github.com/xuannianz/EfficientDet) licensed under the Apache License, Version 2.0
"""

#from utils import inject_tfkeras_modules, init_tfkeras_custom_objects
import functools
import cv2
import numpy as np
import sys
print(sys.executable)
_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None


def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)

def inject_tfkeras_modules(func):
    import tensorflow.keras as tfkeras
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = tfkeras.backend
        kwargs['layers'] = tfkeras.layers
        kwargs['models'] = tfkeras.models
        kwargs['utils'] = tfkeras.utils
        return func(*args, **kwargs)

    return wrapper

def init_tfkeras_custom_objects():
    import tensorflow.keras as tfkeras
    import efficientnet as model

    custom_objects = {
        'swish': inject_tfkeras_modules(model.get_swish)(),
        'FixedDropout': inject_tfkeras_modules(model.get_dropout)()
    }

    tfkeras.utils.get_custom_objects().update(custom_objects)


#from my_efficientnet.efficientnet import EfficientNetB0
import sys
sys.path.append('../')  # Add the parent directory to the Python path

import efficientnet as model
print(model)
#from my_efficientnet import EfficientNet as model
#from my_efficientnet import EfficientNetB0
#print(EfficientNetB0,'hoi')


EfficientNetB0 = inject_tfkeras_modules(model.EfficientNetB0)
EfficientNetB1 = inject_tfkeras_modules(model.EfficientNetB1)
EfficientNetB2 = inject_tfkeras_modules(model.EfficientNetB2)
EfficientNetB3 = inject_tfkeras_modules(model.EfficientNetB3)
EfficientNetB4 = inject_tfkeras_modules(model.EfficientNetB4)
EfficientNetB5 = inject_tfkeras_modules(model.EfficientNetB5)
EfficientNetB6 = inject_tfkeras_modules(model.EfficientNetB6)
EfficientNetB7 = inject_tfkeras_modules(model.EfficientNetB7)

preprocess_input = inject_tfkeras_modules(model.preprocess_input)

init_tfkeras_custom_objects()
