# LifeOS

LifeOS is a personal AI helper designed to remember what matters to you, learn routines, and help with everyday tasks while keeping you in control. It aims to address a common frustration with assistants: they often forget you between chats, and using them means sending text to a model provider.

This build provides a streaming Nemotron chat in a browser and a command-line chat. It can remember useful details between sessions in readable files you can inspect and change. Tools, skills, approvals, redaction, and the morning brief are not implemented yet.

## Run it in 5 minutes

You need Python 3.11 or newer and an API key for Nebius Token Factory.

1. In a terminal, go to the project folder and create an environment:

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install the packages:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and replace the placeholder with your Nebius Token Factory API key:

   ```powershell
   Copy-Item .env.example .env
   ```

   The example uses Nebius by default. For NVIDIA NIM development, edit the three `LLM_*` values to the commented NVIDIA settings.

4. Start LifeOS:

   ```powershell
   uvicorn main:app --reload
   ```

5. Open <http://127.0.0.1:8000> and send a message. To use the command-line chat instead, run `python cli.py`.

Run the tests with:

```powershell
python -m pytest -q
```

## How memory works

LifeOS keeps three plain-text files for the current user: `profile.md` for who you are, `preferences.md` for how you like things done, and `events.md` for dated plans and events. The files are stored in `data/<user>/memory/`. The default user is `demo`; set `LIFEOS_USER` to use another folder name.

Open the Memory panel to read, add, edit, or delete a line. LifeOS suggests possible memories when you choose **End session**; you can edit, reject, or confirm each one. Only items you confirm and choose **Save selected** are written. Duplicate memories are skipped. You can also type `/end` in the CLI and answer yes or no for each suggestion.

The demo starts with fictional sample memories. Files for other users are local data and are not intended to be committed.

## Privacy, in plain words

Your memory files are readable and editable on the machine running LifeOS. The chat and the conversation used to suggest memories are sent to the model provider configured in `.env` so it can answer or suggest what you might want to remember. This build does not yet redact private details. Do not enter passwords, API keys, or other sensitive information.

The app reads API keys from environment variables (or your local `.env` file). Keep `.env` private; it is excluded from Git.

## Built with

Python, FastAPI, the OpenAI Python SDK, NVIDIA Nemotron, and Nebius Token Factory.
