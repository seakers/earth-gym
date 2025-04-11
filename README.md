# Earth Gym

![License](https://img.shields.io/badge/license-MIT-750014.svg)
![py3comp](https://img.shields.io/badge/py3-compatible-306998.svg)

Gym for RL-trained satellite models.

![Demo](docs/demo.gif)

## Requirements
Simple steps to run the code.
1. Install anaconda or miniconda (do not create environment yet).
2. Clone earthGym repository locally.
3. Run `conda env create -f envDroplet.yml` inside the repository to create environment with the correct dependencies. If an error appears regarding agi-stk12 package, do not worry and go to the next step.
4. Install STK 12 in your computer.
5. Run `python -m pip install "<STK installation directory>/bin/AgPythonAPI/agi.stk12-12.9.1-py3-none-any.whl"` (change file name depending on your version) to install agi-stk12.
