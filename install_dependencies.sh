#! /bin/bash

echo "Installing Dependencies"
pythonversion="miniconda3-3.9-4.10.3"
poetryversion="1.1.13"
echo "Check if python version is correct or not. Current python version is: $pythonversion"
echo "Check if poetry version is correct or not. Current poetry version is: $poetryversion"

if command -v pyenv > /dev/null 2>&1; then
    echo "pyenv exists"
else
    echo "pyenv does not exist. Installing pyenv."
	curl https://pyenv.run | bash

	echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
	echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
	echo 'eval "$(pyenv init -)"' >> ~/.bashrc

	$SHELL
fi

export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
pyenv install $pythonversion
pyenv local $pythonversion

# Create local poetry environment
rm -rf .venv
rm -rf poetry.lock
python3 -m venv .venv
./.venv/bin/pip install -U pip setuptools
./.venv/bin/pip install poetry==$poetryversion
POETRY_VIRTUALENVS_IN_PROJECT="true"

# Install Poetry Dependencies
./.venv/bin/poetry

#Test Installation.venv/bin/poetry run which python
./.venv/bin/poetry run python --version
./.venv/bin/poetry install --no-root