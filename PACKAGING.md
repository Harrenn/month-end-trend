# Packaging the Dashboard as a Windows Executable

The project now includes everything required to produce a standalone, no-admin
Windows bundle that launches the dashboard with a single double-click. Because
PyInstaller builds are not cross-platform, the actual `.exe` must be generated
on Windows. You can either run the build locally on Windows or let GitHub
Actions produce the artifact for you.

## Option A: GitHub Actions

1. Push your changes or manually trigger the **Build Windows Bundle** workflow.
2. When it succeeds, download the `MonthEndTrend-windows` artifact from the run.
3. The artifact contains `MonthEndTrend.zip`, ready to distribute.

## Option B: Local Windows build

Follow the steps below on a Windows machine that has Python installed.

### 1. Set up a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller
```

### 2. Build the executable and ZIP bundle

```powershell
python build_release.py
```

The script runs PyInstaller using `run_app.spec`, gathers the necessary data
files, and creates `MonthEndTrend.zip` in the project root. The archive contains
the executable and its isolated workspace so it can run without administrator
rights.

## 3. Distribute

Share `MonthEndTrend.zip`. When extracted, the folder includes:

- `MonthEndTrend.exe` — double-click to launch the dashboard. It opens the
  browser automatically.
- `data/` — working data that users can replace or update through the UI.
- Flask templates and dependencies needed at runtime.

User uploads and edits remain inside this folder, keeping each copy
self-contained.
