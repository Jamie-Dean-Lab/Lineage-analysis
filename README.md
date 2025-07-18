# Lineage-analysis
for private use only; work in progress

# Install dependencies for the python code
Recommend to use `uv` instead of `pip`. Remember to deactivate all virtual environment before the installation.

- For `uv` installation:
```
pip install uv
uv sync
```

For `pip` installtion:
```
python3 -m venv lineage_analysis_env
source lineage_analysis_env/bin/activate
pip install --upgrade pip
pip install -e .
```

# To run the python code
- `uv` installation
```
source .venv/bin/activate
./run-lineage-analysis.py --help
```

- `pip` installation
```
source lineage_analysis_env/bin/activate
./run-lineage-analysis.py --help
```

# Install dependencies for the julia code (ploting the posterior distribution)
```
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

# Plot the posterior distribution
```
julia --project=. -e 'include(\"readandanalysemultipleABCstatistics.jl\"); readandanalysemultipleABCstatistics()'
```






