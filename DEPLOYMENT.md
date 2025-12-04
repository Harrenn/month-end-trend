# Deployment Guide

This project is a Flask web application that can be run with the built-in
development server. For a lightweight deployment you can run the same command
under `systemd` so it restarts automatically if it stops.

## 1. Prerequisites

- Python 3.10 or newer.
- A virtual environment tool such as `python -m venv`.
- Access to systemd (most modern Linux distributions).

## 2. Initial Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Environment variables:

- `FLASK_SECRET_KEY`: set this to a unique, random string in production.
- Optional: `FLASK_DEBUG=1` if you want Flask's debug mode locally.

The application expects CSV files under the `data/` directory. The startup code
creates per-trend subdirectories automatically, but ensure the service user has
read/write access if uploads will be performed.

## 3. Local Development

```bash
# Development server (auto-reloads if FLASK_DEBUG=1)
python app.py
```

## 4. systemd Service

1. Create a service account (recommended):

   ```bash
   sudo useradd --system --create-home --shell /usr/sbin/nologin trend
   sudo chown -R trend:trend /home/trend/trend_web
   ```

2. Create an environment file to store secrets, e.g. `/etc/sysconfig/trend_web`:

   ```
   FLASK_SECRET_KEY=replace-me-with-a-random-string
   ```

3. Create `/etc/systemd/system/trend_web.service` with the contents below
   (update paths to match your checkout location and Python version):

   ```
   [Unit]
   Description=Trend Web Flask Application
   After=network.target

   [Service]
   User=trend
   Group=trend
   WorkingDirectory=/home/trend/trend_web
   EnvironmentFile=/etc/sysconfig/trend_web
   ExecStart=/home/trend/trend_web/.venv/bin/python /home/trend/trend_web/app.py
   Restart=always
   RestartSec=5
   KillSignal=SIGINT
   TimeoutStopSec=30
   PrivateTmp=true

   [Install]
   WantedBy=multi-user.target
   ```

4. Reload systemd and enable the service:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now trend_web.service
   ```

5. Inspect logs:

   ```bash
   journalctl -u trend_web.service -f
   ```

The Flask app honours `FLASK_DEBUG`; leave it unset (or set it to `0`) when running
under systemd so the development reloader stays disabled. `Restart=always` will
bring the service back whenever it exits.

## 5. Updating the Service

```bash
sudo systemctl stop trend_web.service
git pull origin main
source /home/trend/trend_web/.venv/bin/activate
pip install -r requirements.txt
sudo systemctl start trend_web.service
```

Use `sudo systemctl restart trend_web.service` for quick restarts when you only
modify Python code.
