# AGENTS.md

## Start Here

- You are working on Windows in PowerShell. Prefer explicit PowerShell cmdlets such as `Get-ChildItem`, `Set-Location`, `Get-Content`, and `Set-Content`; avoid Unix-only shell assumptions.
- Before editing, inspect the target tree and read repo-local guidance: `AGENTS.md`, `README.md`, `CLAUDE.md`, and `HOWTO.md` when present.
- Keep commands runnable from the repository root unless a section says to `Set-Location` into a project folder.
- This repo contains multiple Deep-Live-Cam variants under `projects/`; do not assume changes in one variant should be copied to another without checking the matching files.

## Repository Map

- `projects/Deep-Live-Cam/` - local Deep-Live-Cam fork with GUI/CLI, webcam, local frame processors, and Colab notebooks.
- `projects/Deep-Live-Cam-Remote/` - remote/batch-oriented fork with `colab_batch.py`, `colab_api.py`, Windows remote app, tests, and the markerized Colab notebook pair.
- `projects/Deep-Live-Cam-Remote/google-colab/Deep_Live_Cam_Remote_Batch.py` - readable markerized notebook source.
- `projects/Deep-Live-Cam-Remote/google-colab/Deep_Live_Cam_Remote_Batch.ipynb` - generated/Colab notebook artifact to keep synchronized with the markerized source.
- `projects/Colab-only-Deep-Live-Cam/` - standalone Colab-only notebook variant.
- `devdocs/` - planning and implementation notes.
- `tmp/` - temporary local artifacts; do not treat as source unless the user points at a specific file.

## Python / Deep-Live-Cam Workflow

- For local app work, use the relevant project directory, usually:
  ```powershell
  Set-Location .\projects\Deep-Live-Cam-Remote
  py -3.11 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install -r requirements.txt
  ```
- GUI entry point: `python run.py` from the chosen Deep-Live-Cam project.
- Batch/remote entry points from `projects/Deep-Live-Cam-Remote`: `python colab_batch.py ...`, `python colab_api.py --host 0.0.0.0 --port 7860`, and `python run_windows_remote_app.py`.
- Focused validation for the remote batch workflow:
  ```powershell
  Set-Location .\projects\Deep-Live-Cam-Remote
  python -m pytest tests\test_colab_batch.py -q
  ```
- For Python syntax checks, prefer `python -m py_compile <file>` on the files you changed.


## Windows Remote App / Colab API

- The Windows app lives in `projects/Deep-Live-Cam-Remote/windows_app/` and launches via `run_windows_remote_app.py`, `run_windows_remote_app.ps1`, or `run-windows-remote-app.bat`.
- Use the project `.venv` for Python package installs and app runs. Example: `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- The Colab API lives in `projects/Deep-Live-Cam-Remote/colab_api.py` and exposes HTTP job endpoints plus WebSocket live/progress endpoints on port `7860` by default.
- The preset Drive layout is `/content/drive/MyDrive/DeepLiveCamRemote/{source,photos,videos,outputs}`. Do not add upload/download workflows unless requested.
- The remote app/backend is intentionally remote-only and does not add NSFW filtering, consent modals, or safety-gate UI.
- Photo batches use `python colab_batch.py photos ...`; video batches use `python colab_batch.py process ...`. Keep output paths mirrored relative to the selected input root.

### Windows App Features
- **Dark title bar** on Windows 10/11 via DWM API; custom app icon in title bar/taskbar.
- **Photos and Videos tabs** both expose full processing options: recursive, overwrite, skip processed, many faces, enhancer, opacity, sharpness, mouth mask, interpolation, poisson blend, color correction.
- **Video percentage range**: start/end % spinboxes to process only a portion of videos.
- **Start/Stop toggle**: batch start buttons switch to red Stop when running; cancel is graceful.
- **Outputs tab**: resizable split view with list panel and preview/player; autoplay with prefetch.
- **Local file upload**: source faces and input folders can be local Windows paths; the app uploads to Colab before starting jobs.
- **Settings sync**: changes in one tab (Photos/Videos) sync to the other when saving or starting jobs.

### Colab Notebook Features
- **Resumable cells**: Clone/install, Tailscale install, and Tailscale auth cells skip already-completed steps.
- **Auto-update on re-run**: Setup cell runs `git pull` when repo already exists, so code updates apply without deleting the cloned directory.

## Notebook Round-Trip

The `Deep_Live_Cam_Remote_Batch.ipynb` Colab notebook uses git clone to fetch the latest code from this repository. Changes to the notebook structure should be made in the markerized `.py` source and rebuilt to `.ipynb`.

**Key change**: The notebook no longer embeds the Python source code as a bundle. Instead, it clones the repository directly from GitHub during the setup cell.

Rules:

- Edit the markerized `.py` source (`Deep_Live_Cam_Remote_Batch.py`) for deterministic diffs; rebuild the `.ipynb` before committing.
- After edits made in Colab or directly in an `.ipynb`, export back to markerized `.py` and review the diff.
- Preserve cell ids, marker lines, `meta_b64`, `NOTEBOOK_META_B64`, `MARKDOWN` / `ENDMARKDOWN`, and `RAW` / `ENDRAW` sentinels.
- Remove throwaway round-trip files such as `_roundtrip.py`, `_roundtrip.ipynb`, or temp notebooks after validation unless the user asks to keep them.
- Run conversions from the repo root when possible, then check `git diff`.

Commands for the remote batch notebook:

```powershell
# Markerized py -> notebook (after editing the .py source)
python scripts/py_to_ipynb.py `
  .\projects\Deep-Live-Cam-Remote\google-colab\Deep_Live_Cam_Remote_Batch.py `
  .\projects\Deep-Live-Cam-Remote\google-colab\Deep_Live_Cam_Remote_Batch.ipynb `
  --eol auto

# Notebook -> markerized py (if edited in Colab)
python scripts/ipynb_to_py.py `
  .\projects\Deep-Live-Cam-Remote\google-colab\Deep_Live_Cam_Remote_Batch.ipynb `
  .\projects\Deep-Live-Cam-Remote\google-colab\Deep_Live_Cam_Remote_Batch.py `
  --eol auto
```

**Important**: Since the notebook clones from GitHub, you must push changes to the main branch before running the notebook in Colab for the first time. The notebook clones from `https://github.com/djebaz/AiSwap.git` branch `main`.

## Context7 Documentation Rule

Use Context7 MCP to fetch current documentation whenever the user asks about a library, framework, SDK, API, CLI tool, or cloud service. Start with `resolve-library-id` unless the user provides an exact `/org/project` library ID, then call `query-docs` with the selected ID and the user's full question. Prefer Context7 over web search for library docs.

Do not use Context7 for general refactoring, writing scripts from scratch, business-logic debugging, code review, or general programming concepts.

## Git / Safety

- Check `git status --short` before and after edits.
- Keep generated artifacts, caches, downloaded models, and temp files out of commits unless explicitly requested.
- Avoid broad repo rewrites; this repository includes upstream forks and generated notebooks.
- When changing notebook-backed workflows, keep `.py` and `.ipynb` synchronized and mention the conversion command used in the handoff.
