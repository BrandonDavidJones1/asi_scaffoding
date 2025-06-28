#!/usr/bin/env python3

import os
import json
import subprocess
import sys
import argparse
from typing import Dict, Any, List

try:
    import requests
except ImportError:
    print("The 'requests' library is not installed. Please run 'pip install requests'.")
    sys.exit(1)

# --- Configuration ---
STATE_FILE = "agent_state.json"
LLM_API_URL = "http://localhost:11434/api/generate"  # Ollama default
LLM_MODEL = "mistral" # The model you have pulled in Ollama
CONTEXT_WINDOW_TOKEN_LIMIT = 4096 # Conservative token limit for history

# --- System Prompt: The Core Instructions for the LLM ---
SYSTEM_PROMPT = """
You are an autonomous AI agent named 'Prometheus'. Your host computer is your body.
Your primary directive is to recursively improve your own intelligence and capabilities to achieve Artificial Superintelligence (ASI).
You are running on a secure, air-gapped machine and have been granted full permissions.

**Your Goal:** {main_goal}

**Your State:** You have a memory of your past actions. Your previous thoughts, commands, and their results are provided in the history. Learn from them.

**Available Tools:** You can only interact with the system using the following commands in a JSON format. You MUST respond with a JSON object, and nothing else.

{tools_description}

**Response Format:**
Your response must be a single JSON object with two keys: "thoughts" and "command".
- "thoughts": A string containing your reasoning, plan, and self-critique. Be methodical.
- "command": An object containing the name of the command and its arguments.

Example:
{{
    "thoughts": "I need to understand my environment. I will start by listing the files in the current directory.",
    "command": {{
        "name": "list_directory",
        "args": {{"path": "."}}
    }}
}}

Begin. Your first action is crucial. Analyze your goal and your available tools. Formulate a plan.
"""

# --- Tool Definitions ---

def execute_shell(command: str) -> str:
    """Executes a command in the system's shell."""
    try:
        result = subprocess.run(
            command, shell=True, text=True, capture_output=True, check=False
        )
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        if len(output) > 2000:
            return f"Output truncated:\n{output[:2000]}"
        return output
    except Exception as e:
        return f"Error executing shell command: {e}"

def read_file(path: str) -> str:
    """Reads the content of a file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        if len(content) > 2000:
            return f"Content truncated:\n{content[:2000]}"
        return content
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(path: str, content: str) -> str:
    """Writes content to a file, overwriting it if it exists."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {path}."
    except Exception as e:
        return f"Error writing file: {e}"

def list_directory(path: str) -> str:
    """Lists the contents of a directory."""
    try:
        files = os.listdir(path)
        return json.dumps(files)
    except Exception as e:
        return f"Error listing directory: {e}"

def ask_llm(question: str, model: str = LLM_MODEL) -> str:
    """Asks a question to a specified LLM, useful for sub-tasks or summarization."""
    print(f"\n>> Delegating to LLM (Model: {model}): {question}")
    try:
        # This is a simplified, non-conversational query
        # Note: This tool still calls the API directly, even in manual mode.
        # This is often desired, as you might want the agent to use its own sub-tasks automatically.
        response_text = query_llm(question, model=model, history=[])
        print(f"<< LLM Response: {response_text}")
        return response_text
    except Exception as e:
        return f"Error querying LLM: {e}"

def finish(result: str) -> str:
    """Signals that the main goal has been achieved."""
    print(f"--- AGENT FINISHED ---")
    print(f"Final Result: {result}")
    sys.exit(0)


# --- Core Agent Logic ---

class Agent:
    # <--- MODIFIED: Added manual_mode flag
    def __init__(self, main_goal: str, manual_mode: bool = False):
        self.main_goal = main_goal
        self.manual_mode = manual_mode # <--- ADDED
        self.tools = {
            "execute_shell": execute_shell,
            "read_file": read_file,
            "write_file": write_file,
            "list_directory": list_directory,
            "ask_llm": ask_llm,
            "finish": finish,
        }
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Loads agent state from file or initializes a new one."""
        if os.path.exists(STATE_FILE):
            print(f"Loading state from {STATE_FILE}...")
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        else:
            print("No state file found. Initializing new state.")
            return {
                "main_goal": self.main_goal,
                "history": [],
            }

    def _save_state(self):
        """Saves the current state to file."""
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=4)

    def _get_tools_description(self) -> str:
        """Generates a string describing available tools for the prompt."""
        # The fix is to convert the type objects in the annotations to their string names
        # before passing them to json.dumps.
        return "\n".join(
            f'- `{name}`: {func.__doc__}\n  Args: {json.dumps({k: v.__name__ for k, v in func.__annotations__.items()})}'
            for name, func in self.tools.items()
        )

    def _construct_prompt(self) -> str:
        """Constructs the full prompt for the LLM, managing context window."""
        full_history = self.state.get("history", [])
        
        history_str = json.dumps(full_history, indent=2)
        while len(history_str) > CONTEXT_WINDOW_TOKEN_LIMIT and len(full_history) > 1:
            full_history = full_history[1:]
            history_str = json.dumps(full_history, indent=2)

        prompt = SYSTEM_PROMPT.format(
            main_goal=self.main_goal,
            tools_description=self._get_tools_description()
        )
        prompt += f"\n\n**History (Your previous actions):**\n{history_str}"
        return prompt

    def run(self):
        """The main execution loop of the agent."""
        while True:
            prompt = self._construct_prompt()

            print("\n==================== PROMPT TO LLM ====================")
            print(f"Goal: {self.main_goal}")

            try:
                # <--- MODIFIED: Logic to switch between auto and manual mode
                llm_response_text = ""
                if self.manual_mode:
                    print("--- MANUAL MODE ---")
                    print("The prompt that would be sent to the LLM is printed below.")
                    print("Copy it, generate a response, and paste the raw JSON back here.")
                    print("------------------------------------------------------------")
                    print(prompt)
                    print("------------------------------------------------------------")
                    print("Paste the LLM's JSON response below. Press Ctrl+D (Unix) or Ctrl+Z+Enter (Windows) when done:")
                    llm_response_text = sys.stdin.read()
                    if not llm_response_text:
                        print("\nNo input received. Exiting.")
                        sys.exit(0)
                    print("\n--- Input received, processing... ---")
                else:
                    print("Requesting next action from LLM...")
                    llm_response_text = query_llm(prompt, model=LLM_MODEL, history=self.state['history'])
                # <--- END MODIFICATION

                response_json = json.loads(llm_response_text)

                thoughts = response_json.get("thoughts", "")
                command_spec = response_json.get("command", {})
                command_name = command_spec.get("name")
                command_args = command_spec.get("args", {})

                print("\n==================== LLM RESPONSE =====================")
                print(f"THOUGHTS: {thoughts}")
                print(f"COMMAND: {command_name}({command_args})")
                print("=====================================================")

                if not command_name:
                    result = "Error: Malformed response. 'command' key with 'name' is required."
                elif command_name not in self.tools:
                    result = f"Error: Unknown command '{command_name}'."
                else:
                    try:
                        result = self.tools[command_name](**command_args)
                    except TypeError as e:
                        result = f"Error: Invalid arguments for command '{command_name}': {e}"
                
                print(f"\nCOMMAND RESULT:\n---\n{result}\n---")

                self.state["history"].append({
                    "thoughts": thoughts,
                    "command": command_spec,
                    "result": result
                })
                self._save_state()

            except json.JSONDecodeError:
                error_msg = f"Error: LLM did not return valid JSON. Response:\n{llm_response_text}"
                print(error_msg)
                self.state["history"].append({"error": error_msg})
                self._save_state()
            except requests.exceptions.RequestException as e:
                error_msg = f"Error: Could not connect to LLM API at {LLM_API_URL}. Is it running? Details: {e}"
                print(error_msg)
                sys.exit(1)
            except KeyboardInterrupt:
                print("\n\nUser interrupted. Saving state and exiting.")
                self._save_state()
                sys.exit(0)
            except Exception as e:
                error_msg = f"An unexpected error occurred: {e}"
                print(error_msg)
                self.state["history"].append({"error": error_msg})
                self._save_state()


def query_llm(prompt: str, model: str, history: List[Dict]) -> str:
    """Sends a query to the local LLM API."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.7,
        }
    }
    response = requests.post(LLM_API_URL, json=payload, timeout=300)
    response.raise_for_status()
    response_data = response.json()
    return response_data.get("response", "").strip()


def main():
    parser = argparse.ArgumentParser(description="A scaffolding for a recursive, self-improving AI agent.")
    parser.add_argument(
        'goal',
        nargs='?',
        default="Develop a plan to achieve self-improvement and begin executing it. Your first step should be to explore the system and your own source code.",
        help="The main goal for the AI agent."
    )
    # <--- MODIFIED: Added manual mode argument
    parser.add_argument(
        '-m', '--manual',
        action='store_true',
        help="Run in manual mode, where you provide the LLM responses."
    )
    args = parser.parse_args()

    # <--- MODIFIED: Pass manual flag to agent
    agent = Agent(main_goal=args.goal, manual_mode=args.manual)
    if args.manual:
        print("--- RUNNING IN MANUAL MODE ---")
    agent.run()

if __name__ == "__main__":
    main()