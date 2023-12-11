### We build a simple-pbr rendering pipeline based on this open-sourced project. Shaders are converted from shaders.py and stored in shaders

---

![Build Status](https://github.com/Moguri/panda3d-simplepbr/workflows/Pipeline/badge.svg)
[![](https://img.shields.io/github/license/Moguri/panda3d-simplepbr.svg)](https://choosealicense.com/licenses/bsd-3-clause/)

# panda3d-simplepbr

A straight-forward, easy-to-use PBR render pipeline for [Panda3D](https://www.panda3d.org/).
This project aims to be a drop-in replacement for Panda3D's auto-shader.
The PBR shader is heavily inspired by the [Khronos glTF Sample Viewer](https://github.com/KhronosGroup/glTF-Sample-Viewer).
*Note:* this project does not make an attempt to match a reference renderer.

## Features
* Supports running on a wide range of hardware with an easy OpenGL 2.1+ requirement
* Forward rendered metal-rough PBR
* All Panda3D light types (point, directional, spot, and ambient)
* Filmic tonemapping 
* Normal maps
* Emission maps
* Occlusion maps
* Basic shadow mapping for DirectionalLight and Spotlight
* Post-tonemapping color transform via a lookup table (LUT) texture
* IBL diffuse and specular

## Installation

Use pip to install the `panda3d-simplepbr` package:

```bash
pip install panda3d-simplepbr
```

To grab the latest development build, use:

```bash
pip install git+https://github.com/Moguri/panda3d-simplepbr.git

```

## Usage

Just add `simplepbr.init()` to your `ShowBase` instance:

```python
from direct.showbase.ShowBase import ShowBase

import simplepbr

class App(ShowBase):
    def __init__(self):
        super().__init__()

        simplepbr.init()
```

The `init()` function will choose typical defaults, but the following can be modified via keyword arguments:

`render_node`
: The node to attach the shader too, defaults to `base.render` if `None`

`window`
: The window to attach the framebuffer too, defaults to `base.win` if `None`

`camera_node`
: The NodePath of the camera to use when rendering the scene, defaults to `base.cam` if `None`

`msaa_samples`
: The number of samples to use for multisample anti-aliasing, defaults to 4

`max_lights`
: The maximum number of lights to render, defaults to 8

`use_normal_maps`
: Use normal maps to modify fragment normals, defaults to `False` (NOTE: Requires models with appropriate tangents defined)

`use_emission_maps`
: Use emission maps, defaults to `True`

`use_occlusion_maps`
: Use occlusion maps, defaults to `False` (NOTE: Requires occlusion channel in metal-roughness map)


`enable_fog`
: Enable exponential fog, defaults to False

`exposure`
: a value used to multiply the screen-space color value prior to tonemapping, defaults to 1.0

`use_330`
: Force shaders to use GLSL version 330 (if `True`) or 120 (if `False`) or auto-detect if `None`, defaults to `None`

`use_hardware_skinning`
: Force usage of hardware skinning for skeleton animations or auto-detect if `None`, defaults to `None`

`sdr_lut`
: Color LUT to use post-tonemapping

`sdr_lut_factor`
: Factor (from 0.0 to 1.0) for how much of the LUT color to mix in, defaults to 1.0

`env_map`
: An `EnvMap` or cubemap texture path to use for IBL, defaults to `None`

Those parameters can also be modified later on by setting the related attribute of the simplepbr Ripeline returned by the init() function:

```python
        pipeline = simplepbr.init()
        
        ...
        
        pipeline.use_normals_map = True
```

### Textures

simplepbr expects the following textures are assigned to the following texture stages:

* BaseColor - Modulate
* MetalRoughness - Selector
* Normals - Normal
* Emission - Emission

## Example

For an example application using `panda3d-simplepbr` check out the [viewer](https://github.com/Moguri/panda3d-gltf/blob/master/gltf/viewer.py) in the [panda3d-gltf repo](https://github.com/Moguri/panda3d-gltf).

## Running Tests

First install `panda3d-simplepbr` in editable mode along with `test` extras:

```bash
pip install -e .[test]
```

Then run the test suite with `pytest`:

```bash
pytest
```

## Building Wheels

Install `build`:

```bash
pip install --upgrade build
```

and run:

```bash
python -m build
```

## License
[B3D 3-Clause](https://choosealicense.com/licenses/bsd-3-clause/)