This folder contains files for the documentation: [https://metadrive-simulator.readthedocs.io/](https://metadrive-simulator.readthedocs.io/).

To build documents locally, please run the following codes:

```
pip install sphinx sphinx_rtd_theme nbsphinx
pip install sphinx-copybutton
conda install -c conda-forge pandoc
cd metadrive/documentation
rm -rf build/ && sphinx-build  source build
```
