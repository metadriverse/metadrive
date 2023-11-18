This folder contains files for the documentation: [https://metadrive-simulator.readthedocs.io/](https://metadrive-simulator.readthedocs.io/).

To build documents locally, please run the following codes:

```
pip install sphinx sphinx_rtd_theme mst-nb
pip install sphinx-copybutton
cd metadrive/documentation
rm -rf build/ && sphinx-build -j 4  source build
```
