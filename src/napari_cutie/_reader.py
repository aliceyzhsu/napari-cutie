"""
This reader receive the path to a .npy file (annotation of a single frame)
The naming of .npy file is supposed to be cellpose-styled.
And return a label layer according to the index number
"""
import re

import numpy as np
import napari
from napari.utils.notifications import show_error


def napari_get_reader(path):
    if isinstance(path, list):
        return None

    if not path.endswith("_seg.npy"):
        show_error('it is not a cellpose_styled seg file')
        return None

    active_layer = napari.current_viewer().layers.selection.active
    if active_layer is None or not isinstance(active_layer, napari.layers.Image):
        show_error('current layer is not an Image layer')
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


def reader_function(path):
    data = np.load(path, allow_pickle=True).item()['masks']
    match = re.search(r'(\d+)_seg.npy', path)
    img = napari.current_viewer().layers.selection.active.data
    labels = np.zeros_like(img)
    index = int(match.groups()[0])
    labels[index] = data
    name = f'initial_annotation_{match.groups()[0]}'
    add_kwargs = {"name": name}

    layer_type = "labels"
    return [(labels, add_kwargs, layer_type)]

