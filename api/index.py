from app import app

# Vercel Python runtime looks for a top-level WSGI app object.
# Exposing the Flask app keeps routes/templates unchanged.
