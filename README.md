# data-frame-agent-langchain

A LangChain-based conversational agent that answers natural-language questions
about weather data stored in a CSV file.  The agent uses
[`create_pandas_dataframe_agent`](https://python.langchain.com/docs/integrations/toolkits/pandas/)
with [ChatOllama](https://python.langchain.com/docs/integrations/chat/ollama/)
(Mistral model) for fully local, privacy-preserving inference.

---

## How it compares to a custom implementation

| Aspect | Custom implementation | This LangChain implementation |
|---|---|---|
| LLM calls | Raw `requests.post` to Ollama REST API | `ChatOllama` handles auth, retries, streaming |
| Agent loop | Manual `while True` + response parsing | `AgentExecutor` (ReAct loop built-in) |
| Tool execution | `exec()` / `eval()` called manually | `PythonREPLTool` managed by the agent |
| Memory | Hand-rolled message list | `ConversationBufferMemory` |
| Error handling | Ad-hoc `try/except` blocks | `handle_parsing_errors=True` in AgentExecutor |

---

## Project structure

```
data-frame-agent-langchain/
├── main.py                  # Entry point – CLI argument parsing & logging setup
├── requirements.txt         # Python dependencies
├── data-set/
│   └── weather-data.csv     # Sample weather dataset (cities × dates)
└── src/
    ├── __init__.py
    ├── config.py            # All settings read from environment variables
    ├── data_loader.py       # CSV → pandas DataFrame
    ├── agent.py             # ChatOllama + create_pandas_dataframe_agent
    └── app.py               # Interactive REPL and single-question modes
```

---

## Prerequisites

### 1. Python 3.11+

```bash
python --version
```

### 2. Ollama with the Mistral model

Install Ollama from <https://ollama.com/download>, then pull the model:

```bash
ollama pull mistral
```

Start the Ollama server (it usually starts automatically):

```bash
ollama serve
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/isameerkhan12/data-frame-agent-langchain.git
cd data-frame-agent-langchain

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the agent

### Interactive mode (default)

```bash
python main.py
```

You will see a welcome prompt.  Type any question about the weather data and
press Enter.  Type `exit` or `quit` to stop.

```
You: Which city had the highest average temperature in summer?
🤖  Based on the data, Phoenix had the highest average temperature during the
    summer months (June–August) at approximately 43 °C.

You: What was the wettest month overall?
🤖  July had the highest total precipitation across all cities.
```

### Single-question mode

```bash
python main.py --question "What is the average humidity across all cities?"
```

### Logging verbosity

```bash
python main.py --log-level DEBUG
```

---

## Configuration

All settings are optional and have sensible defaults.  Override them with
environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `mistral` | Model name |
| `OLLAMA_TEMPERATURE` | `0` | LLM temperature (0 = deterministic) |
| `OLLAMA_MAX_TOKENS` | `2048` | Maximum tokens per response |
| `AGENT_MAX_ITERATIONS` | `10` | Max reasoning steps before giving up |
| `AGENT_VERBOSE` | `false` | Print chain-of-thought steps |
| `AGENT_ALLOW_DANGEROUS_CODE` | `true` | Allow Python eval in the REPL tool |
| `DATA_CSV_PATH` | `data-set/weather-data.csv` | Path to the CSV file |
| `LOG_LEVEL` | `INFO` | Python logging level |

Example:

```bash
OLLAMA_MODEL=llama3 AGENT_VERBOSE=true python main.py
```

---

## Dataset

`data-set/weather-data.csv` contains synthetic daily weather observations for
five US cities (New York, Los Angeles, Chicago, Houston, Phoenix) across 2024.

Columns: `date`, `city`, `temperature_celsius`, `humidity_percent`,
`wind_speed_kmh`, `precipitation_mm`, `weather_condition`, `pressure_hpa`,
`visibility_km`, `uv_index`.

To use your own dataset, set `DATA_CSV_PATH` to its path.

---

## Troubleshooting

**`Connection refused` errors**
Make sure the Ollama server is running: `ollama serve`.

**`model not found` errors**
Pull the model first: `ollama pull mistral`.

**Slow first response**
The first query loads the model into memory; subsequent queries are faster.

**`ModuleNotFoundError`**
Ensure you installed dependencies inside your virtual environment:
`pip install -r requirements.txt`.