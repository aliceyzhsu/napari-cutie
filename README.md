# napari-cutie

[comment]: <> ([![License MIT]&#40;https://img.shields.io/pypi/l/napari-cutie.svg?color=green&#41;]&#40;https://github.com/AliceXuYuanzhen/napari-cutie/raw/main/LICENSE&#41;)

[comment]: <> ([![PyPI]&#40;https://img.shields.io/pypi/v/napari-cutie.svg?color=green&#41;]&#40;https://pypi.org/project/napari-cutie&#41;)

[comment]: <> ([![Python Version]&#40;https://img.shields.io/pypi/pyversions/napari-cutie.svg?color=green&#41;]&#40;https://python.org&#41;)

[comment]: <> ([![tests]&#40;https://github.com/AliceXuYuanzhen/napari-cutie/workflows/tests/badge.svg&#41;]&#40;https://github.com/AliceXuYuanzhen/napari-cutie/actions&#41;)

[comment]: <> ([![codecov]&#40;https://codecov.io/gh/AliceXuYuanzhen/napari-cutie/branch/main/graph/badge.svg&#41;]&#40;https://codecov.io/gh/AliceXuYuanzhen/napari-cutie&#41;)

[comment]: <> ([![napari hub]&#40;https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-cutie&#41;]&#40;https://napari-hub.org/plugins/napari-cutie&#41;)

A video object segmentation tool.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation
Tested on Windows only.

#### requirements:
- Python 3.9+
- PyTorch 1.12+ and corresponding torchvision

#### Install Cutie
```
git clone https://github.com/hkchengrex/Cutie.git
cd Cutie
pip install -e .
cd ..
```

#### Install napari
```
pip install napari[all]
```

#### Install napari-cutie plugin
```
git clone https://github.com/aliceyzhsu/napari-cutie.git
cd napari-cutie
pip install -e .
```
Note that you ara supposed to **git clone**. You don't want to directly download zip from github page since this may cause installation failure.

## Get Started
- `napari` to run napari.
- drag in a tiff stack
- drag in the first frame annotation
  - this annotation should be a .npy file
  - if it is named as `[name][frame_num:03d]_seg.npy`, the plugin can identify which frame the annotation belongs to
  - you are also allowed to load .npy before loading tiff stack. The labels layer generated will be 2d if loaded before tiff stack, 3d if else.
  - tracking from other frames than the first frame is not implemented yet.
- now that we have 1. an Image layer with N frames. 2. a Labels layer which have masks annotation on first frame, we are ready to run cutie tracking.
- Plugins -> Cutie Track(napari-cutie) -> set appropriate parameters -> track

## development log
### 07/19/2024
- basic track code fix, now it works thoroughly as expected.
- add a **shift** checkbox. when check, *the tracking box follows cells' movement*.
  - so now, if you don't check the **shift** checkbox, it is a fixed version of original version.
  - if you check the **shift** checkbox, you can see how the new feature works.
### 07/21/2024
- the naming format requirements for reading .npy mask annotation are more relaxed. Also, you can now load either .tif or .npy first as you want.
  - details can be found in ***Get Started***
- the shift checkbox is replaced by a dropdown list with mode choices (fixed_field, shift_field, atomic)
  - **shift_field** mode is the same as the **shift** mode described in dev log **07/19/2024**
  - **atomic** mode: in this mode, the n-frame video is divided into (n-1) 2-frame sub-video
- done some time experiments. see ***Time test*** below.

> You could find that the shift method doesn't work well. 
> 
> A given processor must receive image with same box size, totally depend on first frame annotation, as input on each frame.
> However, some cells could have intensive changes in its shape and size, so that the box may not cover the whole cell.
> 
> And, when using shift, it tends to mis-track one cell to another.
> 
> The codes work perfectly as expected since I've done lots of checks and experiments.
> The problems described above are all about the "shift" method itself.

## Time test
> data: PhC-C2DH-U373
>
> number of frames: 115
>
> cells being tracked: 4
>
> features: quick movement & large changes on shape and size & no mitosis

### EXP 1-1
- 1 batch (4 cells per batch)
- new *processor* object at each **frame**
- tested on **atomic** mode
- 15.7776, 16.5 s, 7.29 frames/s

### EXP 1-2
- 4 batch (1 cells per batch)
- new *processor* object at each **frame**
- tested on **atomic** mode
- error at 53 frame. reason unknown
- 38.6 s (estimated), 2.98 frames/s

### EXP 2-1
- 1 batch (4 cells per batch)
- new *processor* object for every **batch**
- tested on **shift_field** mode
- 6.82, 6.77 s
### EXP 2-2
- 4 batch (1 cells per batch)
- new *processor* object for every **batch**
- tested on **shift_field** mode
- 17.66, 17.97 s

## License

Distributed under the terms of the [MIT] license,
"napari-cutie" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

## References
- This napari plugin depends mainly on [Cutie]

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[Cutie]: https://github.com/hkchengrex/Cutie

[file an issue]: https://github.com/AliceXuYuanzhen/napari-cutie/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/


