# Security policy

## Supported versions

Only the latest version on the `main` branch receives fixes.

## Reporting a vulnerability

Please **do not open a public issue** for security problems.

Report them privately through GitHub instead:
**Security** tab → **Report a vulnerability**
([direct link](https://github.com/esammostafa9-cloud/resisc45-scene-classification/security/advisories/new)).

Include what you found, how to reproduce it, and the impact you expect. You should get a reply
within 7 days.

## Scope

The parts of this project that process untrusted input are:

- the REST API (`src/resisc45/api.py`), which accepts image uploads
- the Gradio demo (`demo/app.py`)
- dataset archive extraction (`src/resisc45/data.py`)

This project stores no credentials. The Hugging Face token used for publishing lives in the
user's own Hugging Face cache (`hf auth login`), outside the repository.
