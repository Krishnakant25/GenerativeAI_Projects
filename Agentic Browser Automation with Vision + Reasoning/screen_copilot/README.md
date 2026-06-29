# Screen Copilot — Offline AI Knowledge Assistant

A fully offline, privacy-first AI copilot that watches your active
window and provides contextual help — designed for secure, air-gapped,
and sensitive environments where cloud AI tools are not an option.

## Why Offline-First

Unlike cloud-based assistants (ChatGPT, Copilot, Gemini), Screen Copilot
runs entirely on local hardware. No screen content, no selected text,
no context ever leaves your machine. This makes it suitable for:

- **Air-gapped / classified networks** — defense, government, secure facilities
- **Confidential document review** — legal, medical, financial analysis
- **Regulated industries** — where data residency and privacy are mandatory
- **No-internet environments** — field work, secure labs, offline research

## Privacy Guarantees

- 100% local inference via Ollama — zero cloud dependency
- Automatic skipping of sensitive apps (password managers, banking, etc.)
- Local-only audit log for compliance trails
- OCR and reasoning happen entirely on-device

## Features

- **Ambient context awareness** — understands what you're working on
- **Selection analysis** (Ctrl+Alt+S) — highlight text, get focused offline analysis
- **On-demand detail** (Ctrl+Alt+D) — expand any tip for deeper explanation
- **OCR-grounded** — Tesseract extracts exact on-screen text for accuracy
- **App blacklist** — sensitive windows never captured
- **Audit logging** — every action recorded locally

## Architecture

Active Window → OCR (Tesseract) → Vision (llava:7b) →
Context Manager → Reasoning (llama3.1:8b) → Overlay
All local. All private.

## Tradeoffs

This prioritizes privacy and offline operation over speed. On consumer
hardware (e.g. RTX 2060), expect 30-60s per analysis cycle — a deliberate
tradeoff for environments where cloud tools are prohibited entirely.

## Setup

pip install -r requirements.txt
(Tesseract OCR binary required — see https://github.com/UB-Mannheim/tesseract/wiki)

ollama pull llava:7b
ollama pull llama3.1:8b-instruct-q4_0

python -m screen_copilot.main

## Hotkeys
- Ctrl+Alt+S — analyze highlighted selection
- Ctrl+Alt+D — show more detail on current tip
