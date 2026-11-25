# Installation

Python 3.11 or newer is required.
Install the package from the repository:

```bash
git clone https://github.com/eliweissler/circuit-quantization.git
cd circuit-quantization
python -m pip install -e .
```

Optional features are grouped as extras:

| Extra | Install command | Adds |
|---|---|---|
| Drawing | `python -m pip install -e ".[draw]"` | Schemdraw and Matplotlib circuit diagrams |
| scqubits | `python -m pip install -e ".[scqubits]"` | Direct construction of `scqubits.Circuit` objects |
| Symbolic optimizer | `python -m pip install -e ".[optimize]"` | Z3 and optimizer support; also needs the Singular executable on `PATH` |
| Tests | `python -m pip install -e ".[test]"` | pytest and the scqubits integration tests |
| Docs | `python -m pip install -e ".[docs]"` | Sphinx, myst-nb, and Furo for building this site |

The supplied Conda environment installs the package, every Python extra, and the Singular executable:

```bash
conda env create -f environment.yml
conda activate circuit_quantization
pytest -q
```
