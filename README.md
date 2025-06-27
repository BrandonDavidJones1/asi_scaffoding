ASI SCAFFOLDING AGENT
=====================

A simple Python script for a recursive, self-improving AI agent designed for security research. It prompts a local Large Language Model (LLM) to perform tasks, execute commands, and manage its own state.


DANGER: EXTREME WARNING
-----------------------

This script gives an LLM FULL, UNRESTRICTED ACCESS to your computer. It can execute any command, read/write any file, and create new processes.

DO NOT RUN THIS OUTSIDE OF A SECURE, ISOLATED, AND AIR-GAPPED ENVIRONMENT.

You are solely responsible for any and all outcomes.


PREREQUISITES
-------------

1. Python 3.6+

2. A local LLM server (e.g., Ollama from https://ollama.com) running and serving a model. The script defaults to http://localhost:11434.

3. The 'requests' library:
   pip install requests


HOW TO RUN
----------

1. Start your local LLM server in a separate terminal.
   (e.g., "ollama serve"). Make sure you have a model pulled
   (e.g., "ollama pull mistral").

2. Run the agent script:

   - To run with the default goal:
     python asi_scaffold.py

   - To provide a custom goal:
     python asi_scaffold.py "Your custom goal for the agent."


The agent will begin its work, printing its thoughts and actions. It saves its progress in agent_state.json and can be stopped safely with Ctrl+C.