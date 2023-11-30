This folder contains files for the documentation: [https://metadrive-simulator.readthedocs.io/](https://metadrive-simulator.readthedocs.io/).

To build documents locally, please run the following codes:

```
pip install sphinx sphinx_rtd_theme mst-nb
pip install sphinx-copybutton
cd metadrive/documentation
rm -rf build/ && sphinx-build -j 4 source build
```

add `-b linkcheck` for linkcheck. some cross-referenced link might be broken in the check result, which is expected.
This tool can only check some external links. For checking cross-reference, using tools like `linkchecker`.

## Cross-Reference

### How to reference a section of an `.ipynb` file in an RST file?
Supposing you want to reference a section called TopDwnObservation in obs_action.ipynb, use
```
`TopDownObservation <obs_action.html#topdownobservation>`_
```

### How to reference a section of an `.rst` file in an `.ipynb` file?
Supposing you have a section called `Install MetaDrive` in `install.rst`, use
```
 <a href="install.html#install-metadrive">Install MetaDrive</a>
```

### How to reference a section of an `.ipynb` file in an `.ipynb` file?
Supposing you have a section called `Top-down Renderer` in `top_down_render.ipynb`, use 
```
 <a href="top_down_renderer.html#top-down-render">Top-down Renderer</a>
 
 or
 
 [Top-down Renderer](top_down_render.ipynb)
```
The second way can not refer to subtitle.