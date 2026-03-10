# Project Startup Guide

## 1. Environment Setup

Use Python 3.12 for this project.

Open PowerShell and run:

```powershell
cd "D:\Germany\Jobs\working-student\DFKI\ASR-Group\development\data-frame-agent-langchain"

# (Recommended) Create a virtual environment with Python 3.12
py -3.12 -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Ollama & Model Setup

You must have Ollama installed and running locally with the Mistral model:

```powershell
ollama pull mistral
ollama run mistral
ollama serve
```

- Ollama should be accessible at: http://localhost:11434
- If you see connection errors, make sure Ollama is running.

## 3. Running the Agent

### Interactive mode
```powershell
.\.venv\Scripts\python.exe main.py
```
Type your questions about the weather data. Type `exit` or `quit` to stop.

### Single-question mode
```powershell
.\.venv\Scripts\python.exe main.py --question "How many rows does the weather data have?"
```

### Logging verbosity
```powershell
.\.venv\Scripts\python.exe main.py --log-level DEBUG
```

## 4. Configuration (Optional)

You can override defaults with environment variables:

| Variable                  | Default                        | Description                       |
|---------------------------|--------------------------------|-----------------------------------|
| OLLAMA_BASE_URL           | http://localhost:11434         | Ollama server URL                 |
| OLLAMA_MODEL              | mistral                        | Model name                        |
| OLLAMA_TEMPERATURE        | 0                              | LLM temperature                   |
| OLLAMA_MAX_TOKENS         | 2048                           | Max tokens per response           |
| AGENT_MAX_ITERATIONS      | 10                             | Max reasoning steps               |
| AGENT_VERBOSE             | false                          | Print chain-of-thought steps      |
| AGENT_ALLOW_DANGEROUS_CODE| true                           | Allow Python eval in REPL tool    |
| DATA_CSV_PATH             | data-set/weather-data.csv      | Path to the CSV file              |
| LOG_LEVEL                 | INFO                           | Python logging level              |

Example:
```powershell
$env:OLLAMA_MODEL="llama3"
$env:AGENT_VERBOSE="true"
python main.py
```

## 5. Troubleshooting
- If you see `Connection refused`, make sure Ollama is running: `ollama serve`
- If you see `model not found`, pull the model: `ollama pull mistral`
- If you see `ModuleNotFoundError`, ensure dependencies are installed in your venv
- If you see Pydantic errors, use Python 3.11 or 3.12 (not 3.14)

---

For more details, see README.md.
