# LifeOS

LifeOS is a personal AI helper designed to remember what matters to you, learn routines, and help with everyday tasks while keeping you in control. It aims to address a common frustration with assistants: they often forget you between chats, and using them means sending text to a model provider.

This build provides a streaming Nemotron chat, editable file-based memory, and three user-controlled tools: local notes search, task management, and web search. Changes to tasks pause for your approval. Saved skills, private-data redaction, and a morning brief are not implemented yet.

## Run it in 5 minutes

You need Python 3.11 or newer and an API key for NVIDIA NIM or Nebius Token Factory.

1. In a terminal, go to the project folder and create an environment:

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install the packages:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and replace the placeholder with your provider's API key:

   ```powershell
   Copy-Item .env.example .env
   ```

   The example uses NVIDIA NIM and Nemotron. For Nebius, set `LLM_PROVIDER=nebius`, then copy the exact base URL, model name, and settings from the Nebius Token Factory console; the values can vary by model. Keep the key only in `.env`. Web search also needs a Tavily key in `TAVILY_API_KEY`.

4. Start LifeOS:

   ```powershell
   uvicorn main:app --reload
   ```

5. Open <http://127.0.0.1:8000> and send a message. To use the command-line chat instead, run `python cli.py`.

To check the model connection without starting the app, run `python scripts/check_llm.py`.
Reasoning is hidden by default in the web chat; use **Show thinking** to expand it when the model returns reasoning. Memory suggestions and tool-calling requests run with thinking disabled.

Run the tests with:

```powershell
python -m pytest -q
```

## How memory works

LifeOS keeps three plain-text files for the current user: `profile.md` for who you are, `preferences.md` for how you like things done, and `events.md` for dated plans and events. The files are stored in `data/<user>/memory/`. The default user is `demo`; set `LIFEOS_USER` to use another folder name.

Open the Memory panel to read, add, edit, or delete a line. LifeOS suggests possible memories when you choose **End session**; you can edit, reject, or confirm each one. Only items you confirm and choose **Save selected** are written. Duplicate memories are skipped. You can also type `/end` in the CLI and answer yes or no for each suggestion.

The hosted demo is for sample data only. Each visitor gets a separate temporary copy of the fictional demo files; those copies are removed after three hours of inactivity. Do not enter real personal information in the hosted demo.

## What LifeOS can do for you

LifeOS can search text and Markdown notes in your own data folder, list or change your task list, and search the web with Tavily. Notes search returns up to five short snippets. Web search returns up to five results. If Tavily is not configured, LifeOS explains how to enable it instead of crashing. Tool progress appears in the chat as it happens.

## How approvals keep you in control

Reading notes, listing tasks, and web search do not change your files. Adding or completing a task pauses until you see the exact action and choose **Approve** or **Deny**. Approvals expire after two minutes. The CLI asks the same question with a y/n prompt. A denied action is not repeated.

The Activity log records every tool call, its input, whether it was approved, and a short result in `data/<user>/activity.jsonl`. Notes, tasks, and the activity log stay inside that user's data folder. Web search sends its query to Tavily. The conversation is sent to the configured Nemotron provider.

## Privacy, in plain words

Your memory, notes, task list, and activity log are readable files on the machine running LifeOS: your computer when you run it locally, or the server when it is hosted. The hosted demo uses temporary copies of fictional sample data and is for sample data only. Chat and the conversation used to suggest memories are sent to the model provider configured in `.env`. A web search sends its query to Tavily. This build does not yet redact private details. Do not enter passwords, API keys, or other sensitive information.

The app reads API keys from environment variables (or your local `.env` file). Keep `.env` private; it is excluded from Git.

## Built with

Python, FastAPI, the OpenAI Python SDK, NVIDIA Nemotron, and Nebius Token Factory.
