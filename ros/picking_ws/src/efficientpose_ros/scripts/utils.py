from __future__ import division

"""
Source Code from Keras EfficientDet implementation (https://github.com/xuannianz/EfficientDet) licensed under the Apache License, Version 2.0
"""
# Copyright 2019 The TensorFlow Authors, Pavel Yakubovskiy. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

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
    return backend, layers, models, utils


def inject_keras_modules(func):
    import keras
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = keras.backend
        kwargs['layers'] = keras.layers
        kwargs['models'] = keras.models
        kwargs['utils'] = keras.utils
        return func(*args, **kwargs)

    return wrapper


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


def init_keras_custom_objects():
    import keras
    import efficientnet as model

    custom_objects = {
        'swish': inject_keras_modules(model.get_swish)(),
        'FixedDropout': inject_keras_modules(model.get_dropout)()
    }

    keras.utils.generic_utils.get_custom_objects().update(custom_objects)


def init_tfkeras_custom_objects():
    import tensorflow.keras as tfkeras
    import efficientnet as model

    custom_objects = {
        'swish': inject_tfkeras_modules(model.get_swish)(),
        'FixedDropout': inject_tfkeras_modules(model.get_dropout)()
    }

    tfkeras.utils.get_custom_objects().update(custom_objects)


def preprocess_image(image, image_size):
    # image, RGB
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size

    image = cv2.resize(image, (resized_width, resized_height))
    image = image.astype(np.float32)
    image /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image -= mean
    image /= std
    pad_h = image_size - resized_height
    pad_w = image_size - resized_width
    image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')

    return image, scale


def rotate_image(image):
    rotate_degree = np.random.uniform(low=-45, high=45)
    h, w = image.shape[:2]
    # Compute the rotation matrix.
    M = cv2.getRotationMatrix2D(center=(w / 2, h / 2),
                                angle=rotate_degree,
                                scale=1)

    # Get the sine and cosine from the rotation matrix.
    abs_cos_angle = np.abs(M[0, 0])
    abs_sin_angle = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image.
    new_w = int(h * abs_sin_angle + w * abs_cos_angle)
    new_h = int(h * abs_cos_angle + w * abs_sin_angle)

    # Adjust the rotation matrix to take into account the translation.
    M[0, 2] += new_w // 2 - w // 2
    M[1, 2] += new_h // 2 - h // 2

    # Rotate the image.
    image = cv2.warpAffine(image, M=M, dsize=(new_w, new_h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(128, 128, 128))

    return image


def reorder_vertexes(vertexes):
    """
    reorder vertexes as the paper shows, (top, right, bottom, left)
    Args:
        vertexes: np.array (4, 2), should be in clockwise

    Returns:

    """
    assert vertexes.shape == (4, 2)
    xmin, ymin = np.min(vertexes, axis=0)
    xmax, ymax = np.max(vertexes, axis=0)

    # determine the first point with the smallest y,
    # if two vertexes has same y, choose that with smaller x,
    ordered_idxes = np.argsort(vertexes, axis=0)
    ymin1_idx = ordered_idxes[0, 1]
    ymin2_idx = ordered_idxes[1, 1]
    if vertexes[ymin1_idx, 1] == vertexes[ymin2_idx, 1]:
        if vertexes[ymin1_idx, 0] <= vertexes[ymin2_idx, 0]:
            first_vertex_idx = ymin1_idx
        else:
            first_vertex_idx = ymin2_idx
    else:
        first_vertex_idx = ymin1_idx
    ordered_idxes = [(first_vertex_idx + i) % 4 for i in range(4)]
    ordered_vertexes = vertexes[ordered_idxes]
    # drag the point to the corresponding edge
    ordered_vertexes[0, 1] = ymin
    ordered_vertexes[1, 0] = xmax
    ordered_vertexes[2, 1] = ymax
    ordered_vertexes[3, 0] = xmin
    return ordered_vertexes


def postprocess_boxes(boxes, scale, height, width):
    boxes /= scale
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)
    return boxes


import warnings


def label_color(label):
    """ Return a color from a set of predefined colors. Contains 80 colors in total.

    Args
        label: The label to get the color for.

    Returns
        A list of three values representing a RGB color.

        If no color is defined for a certain label, the color green is returned and a warning is printed.
    """
    if label < len(colors):
        return colors[label]
    else:
        warnings.warn('Label {} has no color, returning default.'.format(label))
        return (0, 255, 0)


"""
Generated using:

```
colors = [list((matplotlib.colors.hsv_to_rgb([x, 1.0, 1.0]) * 255).astype(int)) for x in np.arange(0, 1, 1.0 / 80)]
shuffle(colors)
pprint(colors)
```
"""
colors = [
    [255 , 95  , 0]   ,
    [15  , 0   , 255] ,
    [0   , 159 , 255] ,
    #[255 , 19  , 0]   ,
    #[255 , 0   , 0]   ,
    #[255 , 38  , 0]   ,
    #[0   , 255 , 25]  ,
    [255 , 0   , 133] ,
    [255 , 140 , 0]   ,
    #[108 , 0   , 255] ,
    #[0   , 82  , 255] ,
    #[0   , 255 , 6]   ,
    #[255 , 0   , 152] ,
    [223 , 0   , 255] ,
    #[12  , 0   , 255] ,
    [0   , 255 , 255] ,
    #[108 , 255 , 0]   ,
    #[184 , 0   , 255] ,
    #[255 , 0   , 76]  ,
    [245 , 245 , 82]   ,
    [51  , 0   , 255] ,
    [0   , 197 , 255] ,
    [255 , 248 , 0]   ,
    [255 , 0   , 19]  ,
    [255 , 0   , 38]  ,
    #[89  , 255 , 0]   ,
    #[127 , 255 , 0]   ,
    [255 , 153 , 0]   ,
    [0   , 255 , 255] ,
    [0   , 255 , 216] ,
    #[0   , 255 , 121] ,
    [255 , 0   , 248] ,
    [70  , 0   , 255] ,
    [0   , 255 , 159] ,
    [0   , 216 , 255] ,
    [0   , 6   , 255] ,
    [0   , 63  , 255] ,
    #[31  , 255 , 0]   ,
    [255 , 57  , 0]   ,
    [255 , 0   , 210] ,
    #[0   , 255 , 102] ,
    [242 , 255 , 0]   ,
    [255 , 191 , 0]   ,
    #[0   , 255 , 63]  ,
    [255 , 0   , 95]  ,
    [146 , 0   , 255] ,
    [184 , 255 , 0]   ,
    [255 , 114 , 0]   ,
    [0   , 255 , 235] ,
    [255 , 229 , 0]   ,
    [0   , 178 , 255] ,
    [255 , 0   , 114] ,
    [255 , 0   , 57]  ,
    [0   , 140 , 255] ,
    [0   , 121 , 255] ,
    #[12  , 255 , 0]   ,
    [255 , 210 , 0]   ,
    #[0   , 255 , 44]  ,
    [165 , 255 , 0]   ,
    [0   , 25  , 255] ,
    [0   , 255 , 140] ,
    [0   , 101 , 255] ,
    #[0   , 255 , 82]  ,
    [223 , 255 , 0]   ,
    [242 , 0   , 255] ,
    [89  , 0   , 255] ,
    [165 , 0   , 255] ,
    #[70  , 255 , 0]   ,
    [255 , 0   , 172] ,
    [255 , 76  , 0]   ,
    [203 , 255 , 0]   ,
    [204 , 0   , 255] ,
    [255 , 0   , 229] ,
    [255 , 133 , 0]   ,
    [127 , 0   , 255] ,
    [0   , 235 , 255] ,
    [0   , 255 , 197] ,
    [255 , 0   , 191] ,
    [0   , 44  , 255] ,
    #[50  , 255 , 0]
]

"""
EfficientPose (c) by Steinbeis GmbH & Co. KG für Technologietransfer
Haus der Wirtschaft, Willi-Bleicher-Straße 19, 70174 Stuttgart, Germany
Yannick Bukschat: yannick.bukschat@stw.de
Marcus Vetter: marcus.vetter@stw.de

EfficientPose is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

The license can be found in the LICENSE file in the root directory of this source tree
or at http://creativecommons.org/licenses/by-nc/4.0/.
---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------

Based on:

Keras EfficientDet implementation (https://github.com/xuannianz/EfficientDet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
The official EfficientDet implementation (https://github.com/google/automl) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
EfficientNet Keras implementation (https://github.com/qubvel/efficientnet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
Keras RetinaNet implementation (https://github.com/fizyr/keras-retinanet) licensed under the Apache License, Version 2.0
"""

import numpy as np
from tensorflow import keras

#from compute_overlap import compute_overlap


class AnchorParameters:
    """
    The parameters that define how anchors are generated.

    Args
        sizes : List of sizes to use. Each size corresponds to one feature level.
        strides : List of strides to use. Each stride correspond to one feature level.
        ratios : List of ratios to use per location in a feature map.
        scales : List of scales to use per location in a feature map.
    """

    def __init__(self,
                 sizes = (32, 64, 128, 256, 512),
                 strides = (8, 16, 32, 64, 128),
                 ratios = (1, 0.5, 2),
                 scales = (2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.))):
        self.sizes = sizes
        self.strides = strides
        self.ratios = np.array(ratios, dtype=keras.backend.floatx())
        self.scales = np.array(scales, dtype=keras.backend.floatx())

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


"""
The default anchor parameters.
"""
AnchorParameters.default = AnchorParameters(
    sizes = [32, 64, 128, 256, 512],
    strides = [8, 16, 32, 64, 128],
    ratios = np.array([1, 0.5, 2], keras.backend.floatx()),
    scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
)


def anchor_targets_bbox(
        anchors,
        image_group,
        annotations_group,
        num_classes,
        num_rotation_parameters,
        num_translation_parameters,
        translation_anchors,
        negative_overlap = 0.4,
        positive_overlap = 0.5,
):
    """
    Generate anchor targets for bbox detection.

    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        image_group: List of BGR images.
        annotations_group: List of annotations (np.array of shape (N, 5) for (x1, y1, x2, y2, label)).
        num_classes: Number of classes to predict.
        num_rotation_parameters: Number of rotation parameters to regress (e.g. 3 for axis angle representation).
        num_translation_parameters: Number of translation parameters to regress (usually 3).
        translation_anchors: np.array of annotations of shape (N, 2) for (x, y).
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        labels_batch: batch that contains labels & anchor states (np.array of shape (batch_size, N, num_classes + 1),
                      where N is the number of anchors for an image and the last column defines the anchor state
                      (-1 for ignore, 0 for bg, 1 for fg).
        regression_batch: batch that contains bounding-box regression targets for an image & anchor states
                      (np.array of shape (batch_size, N, 4 + 1), where N is the number of anchors for an image,
                      the first 4 columns define regression targets for (x1, y1, x2, y2) and the last column defines
                      anchor states (-1 for ignore, 0 for bg, 1 for fg).
        transformation_batch: batch that contains 6D pose regression targets for an image & anchor states
                      (np.array of shape (batch_size, N, num_rotation_parameters + num_translation_parameters + 1),
                      where N is the number of anchors for an image,
                      the first num_rotation_parameters columns define regression targets for the rotation,
                      the next num_translation_parameters columns define regression targets for the translation,
                      and the last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).
    """

    assert (len(image_group) == len(annotations_group)), "The length of the images and annotations need to be equal."
    assert (len(annotations_group) > 0), "No data received to compute anchor targets for."
    for annotations in annotations_group:
        assert ('bboxes' in annotations), "Annotations should contain bboxes."
        assert ('labels' in annotations), "Annotations should contain labels."
        assert('transformation_targets' in annotations), "Annotations should contain transformation_targets."

    batch_size = len(image_group)

    regression_batch = np.zeros((batch_size, anchors.shape[0], 4 + 1), dtype=np.float32)
    labels_batch = np.zeros((batch_size, anchors.shape[0], num_classes + 1), dtype=np.float32)
    transformation_batch  = np.zeros((batch_size, anchors.shape[0], num_rotation_parameters + num_translation_parameters + 1), dtype = np.float32)

    # compute labels and regression targets
    for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
        if annotations['bboxes'].shape[0]:
            # obtain indices of gt annotations with the greatest overlap
            # argmax_overlaps_inds: id of ground truth box has greatest overlap with anchor
            # (N, ), (N, ), (N, ) N is num_anchors
            positive_indices, ignore_indices, argmax_overlaps_inds = compute_gt_annotations(anchors,
                                                                                            annotations['bboxes'],
                                                                                            negative_overlap,
                                                                                            positive_overlap)
            labels_batch[index, ignore_indices, -1] = -1
            labels_batch[index, positive_indices, -1] = 1

            regression_batch[index, ignore_indices, -1] = -1
            regression_batch[index, positive_indices, -1] = 1
            
            transformation_batch[index, ignore_indices, -1]   = -1
            transformation_batch[index, positive_indices, -1] = 1

            # compute target class labels
            labels_batch[index, positive_indices, annotations['labels'][argmax_overlaps_inds[positive_indices]].astype(int)] = 1

            regression_batch[index, :, :4] = bbox_transform(anchors, annotations['bboxes'][argmax_overlaps_inds, :])
                
            transformation_batch[index, :, :-1] = annotations['transformation_targets'][argmax_overlaps_inds, :]
                
        # ignore anchors outside of image
        if image.shape:
            anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
            indices = np.logical_or(anchors_centers[:, 0] >= image.shape[1], anchors_centers[:, 1] >= image.shape[0])

            labels_batch[index, indices, -1] = -1
            regression_batch[index, indices, -1] = -1
            transformation_batch[index, indices, -1] = -1

    return labels_batch, regression_batch, transformation_batch


def compute_gt_annotations(
        anchors,
        annotations,
        negative_overlap=0.4,
        positive_overlap=0.5
):
    """
    Obtain indices of gt annotations with the greatest overlap.

    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        annotations: np.array of shape (K, 5) for (x1, y1, x2, y2, label).
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        positive_indices: indices of positive anchors, (N, )
        ignore_indices: indices of ignored anchors, (N, )
        argmax_overlaps_inds: ordered overlaps indices, (N, )
    """
    # (N, K)
    overlaps = compute_overlap(anchors.astype(np.float64), annotations.astype(np.float64))
    # (N, )
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    # (N, )
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

    # assign "dont care" labels
    # (N, )
    positive_indices = max_overlaps >= positive_overlap
    
    #get the max overlapping anchor indices for each gt box and set them to true so that each gt box has at least one positive anchor box
    max_overlapping_anchor_box_indices = np.argmax(overlaps, axis = 0)
    positive_indices[max_overlapping_anchor_box_indices] = True

    # (N, )
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, argmax_overlaps_inds


def layer_shapes(image_shape, model):
    """
    Compute layer shapes given input image shape and the model.

    Args
        image_shape: The shape of the image.
        model: The model to use for computing how the image shape is transformed in the pyramid.

    Returns
        A dictionary mapping layer names to image shapes.
    """
    shape = {
        model.layers[0].name: (None,) + image_shape,
    }

    for layer in model.layers[1:]:
        nodes = layer._inbound_nodes
        for node in nodes:
            input_shapes = [shape[inbound_layer.name] for inbound_layer in node.inbound_layers]
            if not input_shapes:
                continue
            shape[layer.name] = layer.compute_output_shape(input_shapes[0] if len(input_shapes) == 1 else input_shapes)

    return shape


def make_shapes_callback(model):
    """
    Make a function for getting the shape of the pyramid levels.
    """

    def get_shapes(image_shape, pyramid_levels):
        shape = layer_shapes(image_shape, model)
        image_shapes = [shape["P{}".format(level)][1:3] for level in pyramid_levels]
        return image_shapes

    return get_shapes


def guess_shapes(image_shape, pyramid_levels):
    """
    Guess shapes based on pyramid levels.

    Args
          image_shape: The shape of the image.
          pyramid_levels: A list of what pyramid levels are used.

    Returns
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels = None,
    anchor_params = None,
    shapes_callback = None,
):
    """ Generators anchors for a given shape.

    Args
        image_shape: The shape of the image.
        pyramid_levels: List of ints representing which pyramids to use (defaults to [3, 4, 5, 6, 7]).
        anchor_params: Struct containing anchor parameters. If None, default values are used.
        shapes_callback: Function to call for getting the shape of the image at different pyramid levels.

    Returns
        anchors np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates for the anchors.
        translation anchors np.array of shape (N, 3) containing the (x, y, stride) coordinates for the anchors.
    """

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    if shapes_callback is None:
        shapes_callback = guess_shapes
    image_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    all_translation_anchors = np.zeros((0, 3))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales
        )
        translation_anchors = np.zeros(shape = (len(anchor_params.ratios) * len(anchor_params.scales), 2))
        shifted_anchors = shift(image_shapes[idx], anchor_params.strides[idx], anchors)
        shifted_translation_anchors = translation_shift(image_shapes[idx], anchor_params.strides[idx], translation_anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)
        all_translation_anchors = np.append(all_translation_anchors, shifted_translation_anchors, axis = 0)

    return all_anchors.astype(np.float32), all_translation_anchors.astype(np.float32)


def shift(feature_map_shape, stride, anchors):
    """
    Produce shifted anchors based on shape of the map and stride size.

    Args
        feature_map_shape : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, feature_map_shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, feature_map_shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def translation_shift(shape, stride, translation_anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        translation_anchors: The translation anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
    )).transpose()

    # add A anchors (1, A, 2) to
    # cell K shifts (K, 1, 2) to get
    # shift anchors (K, A, 2)
    # reshape to (K*A, 2) shifted anchors
    A = translation_anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (translation_anchors.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 2))
    
    #append stride to anchors
    stride_array = np.full((all_anchors.shape[0], 1), stride)
    all_anchors = np.concatenate([all_anchors, stride_array], axis = -1)

    return all_anchors


def generate_anchors(base_size = 16, ratios = None, scales = None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X scales w.r.t. a reference window.

    Args:
        base_size: The base size of the anchor boxes
        ratios: Tuple containing the aspect ratios of the anchor boxes
        scales: Tuple containing the scales of the anchor boxes

    Returns:
        anchors: numpy array (num_anchors, 4) containing the anchor boxes (x1, y1, x2, y2) created with the given size, scales and ratios
    """
    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = base_size * np.tile(np.repeat(scales, len(ratios))[None], (2, 1)).T

    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.tile(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.tile(ratios, len(scales))

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def bbox_transform(anchors, gt_boxes, scale_factors = None):
    """
    Computes the 2D bbox regression targets using the anchor boxes and the grdound truth 2D bounding boxes

    Args:
        anchors: np.array of anchor boxes with shape (N, 4) for (x1, y1, x2, y2).
        gt_boxes: np.array of the ground truth 2D bounding boxes with shape (N, 4) for (x1, y1, x2, y2)
        scale_factors: Optional scale factors

    Returns:
        targets: numpy array (N, 4) containing the 2D bounding box targets
    """
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    cxa = anchors[:, 0] + wa / 2.
    cya = anchors[:, 1] + ha / 2.

    w = gt_boxes[:, 2] - gt_boxes[:, 0]
    h = gt_boxes[:, 3] - gt_boxes[:, 1]
    cx = gt_boxes[:, 0] + w / 2.
    cy = gt_boxes[:, 1] + h / 2.
    # Avoid NaN in division and log below.
    ha += 1e-7
    wa += 1e-7
    h += 1e-7
    w += 1e-7
    tx = (cx - cxa) / wa
    ty = (cy - cya) / ha
    tw = np.log(w / wa)
    th = np.log(h / ha)
    if scale_factors:
        ty /= scale_factors[0]
        tx /= scale_factors[1]
        th /= scale_factors[2]
        tw /= scale_factors[3]
    targets = np.stack([ty, tx, th, tw], axis = 1)
    return targets


def translation_transform(translation_anchors, gt_translations, scale_factors = None):
    """
    Computes the translation regression targets for an image using the translation anchors and ground truth translations.

    Args:
        translation_anchors: np.array of translation anchors with shape (N, 3) for (x, y, stride).
        gt_translations: np.array of the ground truth translations with shape (N, 3) for (x_2D, y_2D, z_3D)
        scale_factors: Optional scale factors

    Returns:
        targets: numpy array (N, 3) containing the translation regression targets
    """

    strides  = translation_anchors[:, -1]

    targets_dx = (gt_translations[:, 0] - translation_anchors[:, 0]) / strides
    targets_dy = (gt_translations[:, 1] - translation_anchors[:, 1]) / strides
    
    if scale_factors:
        targets_dx /= scale_factors[0]
        targets_dy /= scale_factors[1]
    
    targets_tz = gt_translations[:, 2]

    targets = np.stack((targets_dx, targets_dy, targets_tz), axis = 1)

    return targets

"""
EfficientPose (c) by Steinbeis GmbH & Co. KG für Technologietransfer
Haus der Wirtschaft, Willi-Bleicher-Straße 19, 70174 Stuttgart, Germany
Yannick Bukschat: yannick.bukschat@stw.de
Marcus Vetter: marcus.vetter@stw.de

EfficientPose is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

The license can be found in the LICENSE file in the root directory of this source tree
or at http://creativecommons.org/licenses/by-nc/4.0/.
---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------

Based on:

Keras EfficientDet implementation (https://github.com/xuannianz/EfficientDet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
The official EfficientDet implementation (https://github.com/google/automl) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
EfficientNet Keras implementation (https://github.com/qubvel/efficientnet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
Keras RetinaNet implementation (https://github.com/fizyr/keras-retinanet) licensed under
    
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cv2
import numpy as np


def draw_box(image, box, color, thickness = 2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def draw_boxes(image, boxes, color, thickness = 2):
    """ Draws boxes on an image with a given color.

    # Arguments
        image     : The image to draw on.
        boxes     : A [N, 4] matrix (x1, y1, x2, y2).
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    for b in boxes:
        draw_box(image, b, color, thickness=thickness)
        
def draw_bbox_8_2D(draw_img, bbox_8_2D, color = (0, 255, 0), thickness = 2):
    """ Draws the 2D projection of a 3D model's cuboid on an image with a given color.

    # Arguments
        draw_img     : The image to draw on.
        bbox_8_2D    : A [8 or 9, 2] matrix containing the 8 corner points (x, y) and maybe also the centerpoint.
        color     : The color of the boxes.
        thickness : The thickness of the lines to draw boxes with.
    """
    #convert bbox to int and tuple
    bbox = np.copy(bbox_8_2D).astype(np.int32)
    bbox = tuple(map(tuple, bbox))
    
    #lower level
    cv2.line(draw_img, bbox[0], bbox[1], color, thickness)
    cv2.line(draw_img, bbox[1], bbox[2], color, thickness)
    cv2.line(draw_img, bbox[2], bbox[3], color, thickness)
    cv2.line(draw_img, bbox[0], bbox[3], color, thickness)
    #upper level
    cv2.line(draw_img, bbox[4], bbox[5], color, thickness)
    cv2.line(draw_img, bbox[5], bbox[6], color, thickness)
    cv2.line(draw_img, bbox[6], bbox[7], color, thickness)
    cv2.line(draw_img, bbox[4], bbox[7], color, thickness)
    #sides
    cv2.line(draw_img, bbox[0], bbox[4], color, thickness)
    cv2.line(draw_img, bbox[1], bbox[5], color, thickness)
    cv2.line(draw_img, bbox[2], bbox[6], color, thickness)
    cv2.line(draw_img, bbox[3], bbox[7], color, thickness)
    
    #check if centerpoint is also available to draw
    if len(bbox) == 9:
        #draw centerpoint
        cv2.circle(draw_img, bbox[8], 3, color, -1)
    
    
def project_bbox_3D_to_2D(points_bbox_3D, rotation_vector, translation_vector, camera_matrix, append_centerpoint = True):
    """ Projects the 3D model's cuboid onto a 2D image plane with the given rotation, translation and camera matrix.

    Arguments:
        points_bbox_3D: numpy array with shape (8, 3) containing the 8 (x, y, z) corner points of the object's 3D model cuboid 
        rotation_vector: numpy array containing the rotation vector with shape (3,)
        translation_vector: numpy array containing the translation vector with shape (3,)
        camera_matrix: numpy array with shape (3, 3) containing the intrinsic camera parameters
        append_centerpoint: Boolean indicating wheter to append the centerpoint or not
    Returns:
        points_bbox_2D: numpy array with shape (8 or 9, 2) with the 2D projections of the object's 3D cuboid
    """
    print("hh")
    if append_centerpoint:
        points_bbox_3D = np.concatenate([points_bbox_3D, np.zeros(shape = (1, 3))], axis = 0)
    
    points_bbox_2D, jacobian = cv2.projectPoints(points_bbox_3D, rotation_vector, translation_vector, camera_matrix, None)
    points_bbox_2D = np.squeeze(points_bbox_2D)
    
    return points_bbox_2D
    


def draw_detections(image, boxes, scores, labels, rotations, translations, class_to_bbox_3D, camera_matrix, color = None, label_to_name = None, score_threshold = 0.5, draw_bbox_2d = False, draw_name = False):
    """ Draws detections in an image.

    # Arguments
        image: The image to draw on.
        boxes: A [N, 4] matrix (x1, y1, x2, y2).
        scores: A list of N classification scores.
        labels: A list of N labels.
        rotations: A list of N rotations
        translations: A list of N translations
        class_to_bbox_3D: A dictionary mapping the class labels to the object's 3D bboxes (cuboids)
        camera_matrix: numpy array with shape (3, 3) containing the intrinsic camera parameters
        color: The color of the boxes. By default the color from utils.colors.label_color will be used.
        label_to_name: (optional) Functor or dictionary for mapping a label to a name.
        score_threshold : Threshold used for determining what detections to draw.
        draw_bbox_2d: Boolean indicating wheter to draw the 2D bounding boxes or not
        draw_name: Boolean indicating wheter to draw the class names or not
    """
    selection = np.where(scores > score_threshold)[0]
    print("hall0")
    for i in selection:
        if color is None:
            c = label_color(int(labels[i]))
        if draw_bbox_2d:
            draw_box(image, boxes[i, :], color = c)
        translation_vector = translations[i, :]
        points_bbox_2D = project_bbox_3D_to_2D(class_to_bbox_3D[labels[i]], rotations[i, :], translation_vector, camera_matrix, append_centerpoint = True)
        draw_bbox_8_2D(image, points_bbox_2D, color = c)
        if draw_name:
            if isinstance(label_to_name, dict):
                name = label_to_name[labels[i]] if label_to_name else labels[i]
            else:
                name = label_to_name(labels[i]) if label_to_name else labels[i]
            caption = name + ': {0:.2f}'.format(scores[i])
            draw_caption(image, boxes[i, :], caption)


def draw_annotations(image, annotations, class_to_bbox_3D, camera_matrix, color = (0, 255, 0), label_to_name = None, draw_bbox_2d = False, draw_name = False):
    """ Draws annotations in an image.

    # Arguments
        image: The image to draw on.
        annotations: A [N, 5] matrix (x1, y1, x2, y2, label) or dictionary containing bboxes (shaped [N, 4]) and labels (shaped [N]) and rotations (shaped [N, 3]) and translations (shaped [N, 4]).
        class_to_bbox_3D: A dictionary mapping the class labels to the object's 3D bboxes (cuboids)
        camera_matrix: numpy array with shape (3, 3) containing the intrinsic camera parameters
        color: The color of the boxes. By default the color from utils.colors.label_color will be used.
        label_to_name: (optional) Functor or dictionary for mapping a label to a name.
        draw_bbox_2d: Boolean indicating wheter to draw the 2D bounding boxes or not
        draw_name: Boolean indicating wheter to draw the class names or not
    """
    if isinstance(annotations, np.ndarray):
        annotations = {'bboxes': annotations[:, :4], 'labels': annotations[:, 4]}

    assert('bboxes' in annotations)
    assert('labels' in annotations)
    assert('rotations' in annotations)
    assert('translations' in annotations)
    assert(annotations['bboxes'].shape[0] == annotations['labels'].shape[0])

    for i in range(annotations['bboxes'].shape[0]):
        label   = annotations['labels'][i]
        if color is None:
            color = (0, 255, 0)
        if draw_bbox_2d:
            draw_box(image, annotations['bboxes'][i], color = (0, 127, 0))
        caption = '{}'.format(label_to_name(label) if label_to_name else label)
        points_bbox_2D = project_bbox_3D_to_2D(class_to_bbox_3D[annotations["labels"][i]], annotations['rotations'][i, :3], annotations['translations'][i, :], camera_matrix, append_centerpoint = True)
        draw_bbox_8_2D(image, points_bbox_2D, color = color)
        if draw_name:
            if isinstance(label_to_name, dict):
                caption = label_to_name[int(label)] if label_to_name else int(label)
            else:
                caption = label_to_name(int(label)) if label_to_name else int(label)
            draw_caption(image, annotations['bboxes'][i], caption)


"""
EfficientPose (c) by Steinbeis GmbH & Co. KG für Technologietransfer
Haus der Wirtschaft, Willi-Bleicher-Straße 19, 70174 Stuttgart, Germany
Yannick Bukschat: yannick.bukschat@stw.de
Marcus Vetter: marcus.vetter@stw.de

EfficientPose is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

The license can be found in the LICENSE file in the root directory of this source tree
or at http://creativecommons.org/licenses/by-nc/4.0/.
---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------

Based on:

Keras EfficientDet implementation (https://github.com/xuannianz/EfficientDet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
The official EfficientDet implementation (https://github.com/google/automl) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
EfficientNet Keras implementation (https://github.com/qubvel/efficientnet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
Keras RetinaNet implementation (https://github.com/fizyr/keras-retinanet) licensed under
    
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import cv2
from PIL import Image


def read_image_bgr(path):
    """
    Read an image in BGR format.

    Args
        path: Path to the image.
    """
    # We deliberately don't use cv2.imread here, since it gives no feedback on errors while reading the image.
    image = np.asarray(Image.open(path).convert('RGB'))
    return image[:, :, ::-1].copy()


def preprocess_image_2(x, mode='caffe'):
    """
    Preprocess an image by subtracting the ImageNet mean.

    Args
        x: np.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.

    Returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already

    # covert always to float32 to keep compatibility with opencv
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x



def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    """
    Compute an image scale such that the image size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resizing scale.
    """
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def resize_image(img, min_side=800, max_side=1333):
    """
    Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    # compute scale to resize the image
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale


def _uniform(val_range):
    """
    Uniformly sample from the given range.

    Args
        val_range: A pair of lower and upper bound.
    """
    return np.random.uniform(val_range[0], val_range[1])


def _check_range(val_range, min_val=None, max_val=None):
    """
    Check whether the range is a valid range.

    Args
        val_range: A pair of lower and upper bound.
        min_val: Minimal value for the lower bound.
        max_val: Maximal value for the upper bound.
    """
    if val_range[0] > val_range[1]:
        raise ValueError('interval lower bound > upper bound')
    if min_val is not None and val_range[0] < min_val:
        raise ValueError('invalid interval lower bound')
    if max_val is not None and val_range[1] > max_val:
        raise ValueError('invalid interval upper bound')


def _clip(image):
    """
    Clip and convert an image to np.uint8.

    Args
        image: Image to clip.
    """
    return np.clip(image, 0, 255).astype(np.uint8)
