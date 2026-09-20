uv run mcp install main.py 


pip install uv

if you are arating the freash project


uv init <project_name>

go inside of project
===================

cd <project_name>


create a env

uv venv

or 

conda create -n <env name> python=3.11 -y

or 
python -m venv venv



activate the env (MAC/Linux)

source  .venv/bin/activate

for window

.venv\Scripts\activate



install Library

uv sync


To implement a basic mockup MCP client (without involving an LLM yet), we will need to install a few core dependencies


uv add mcp asyncio
