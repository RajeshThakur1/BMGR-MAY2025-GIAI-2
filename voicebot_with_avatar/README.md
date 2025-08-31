created the Virtualenv 

```virtualenv  venv```

create a env with conda

```conda create -n <envName> python=3.11 -y```

Install all the Livekit plugins

```commandline

pip install "livekit-agents[openai,cartesia,silero,turn-detector]~=1.0" "livekit-plugins-noise-cancellation~=0.2" "python-dotenv"


```


To Download the required plugins

```commandline
python agent.py download-files

```