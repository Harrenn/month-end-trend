# Project Overview

This is a Flask web application designed for time-series trend analysis. The application provides a web interface to upload, analyze, and visualize data from CSV files. The primary focus is on a custom "Legacy RMLA" model for trend prediction.

**Key Technologies:**

*   **Backend:** Python, Flask
*   **Data Manipulation:** Pandas, NumPy
*   **Frontend:** HTML, with server-side rendering using Flask's templating engine.

**Architecture:**

The application follows a monolithic architecture. The main components are:

*   `app.py`: The core Flask application, containing all the routes, API endpoints, and data analysis logic.
*   `templates/`: HTML templates for the web interface.
*   `data/`:  Directory where the data files (CSV) are stored.

# Building and Running

**1. Prerequisites:**

*   Python 3.10 or newer
*   A virtual environment tool like `venv`

**2. Setup:**

```bash
# Create and activate a virtual environment
python -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**3. Running the Application:**

```bash
# Set the secret key for Flask sessions
export FLASK_SECRET_KEY='a-secret-key'

# Run the development server
python app.py
```

The application will be available at `http://127.0.0.1:5002`.

**4. Deployment:**

The `DEPLOYMENT.md` file provides detailed instructions for deploying the application using `systemd` on a Linux server.

# Development Conventions

*   **Data:** The application expects CSV files in the `data/` directory. The `TREND_CONFIG` in `app.py` defines which subdirectories and files are used.
*   **Model:** The trend prediction logic is implemented in the `get_historical_data` and `api_live_trend` functions in `app.py`.
*   **Configuration:** The application is configured through environment variables (e.g., `FLASK_SECRET_KEY`).
*   **Dependencies:** Python dependencies are managed in the `requirements.txt` file.