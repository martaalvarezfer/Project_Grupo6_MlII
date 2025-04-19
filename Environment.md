# Python Environment Setup

## Using uv (alternative to pip)

more information here: https://github.com/astral-sh/uv

1. First, we need to install uv. This can be done through pip directly.
```bash 
pip install uv
```
2. Next, we'll create the virtual environment named dl_lab for the deep learning lab using python 3.12,as example
```bash 
# uv venv --python=<python_version> <virtual_env_name> 
uv venv --python=3.12 dl_lab
```
3. Let's activate this environment
```bash 
source dl_lab/bin/activate # macos /linux
# OR
dl_lab\Scripts\activate # Windows
```
4. Installing our dependencies
```bash
uv pip install -r requirements.txt
```

## Using conda

1. Check if conda is installed
2. Next, we'll create the virtual environment named dl_lab for the deep learning lab using python 3.12, as example
```bash
conda create -n dl_lab python=3.12
```
3. Let's activate this environment
```bash
conda activate dl_lab
```
4. Installing our dependencies
```bash
pip install -r requirements.txt
```