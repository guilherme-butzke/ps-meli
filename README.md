# ps-mercadolibre

Technical challenge solution developed for Mercado Libre.

## Environment Setup

To execute the code, it is necessary to create the environment.  
The project uses **pyproject.toml** to manage dependencies and packaging (instead of the traditional `requirements.txt`).

To create the environment, follow the commands below for Windows or Linux.  
The prerequisite is having **Python 3.12 installed**.

### Windows

```bash
py -3.12 -m venv .ps_mercado_libre
.ps_mercado_libre\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e .[dev]
```

### Linux / macOS
```bash
py -3.12 -m venv .ps_mercado_libre
source .ps_mercado_libre/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

**Note:**
On some systems, the py command may be different (e.g., python or python3.12 instead of py).
Use python --version or py --version to check which command is available.

## VSCode Configuration (recommended)

To ensure notebooks and scripts run from the project root and use the correct environment, open the **project folder** in VSCode (not individual files).

Create `.vscode/settings.json`:

```json
{
    "jupyter.notebookFileRoot": "${workspaceFolder}"
}
```