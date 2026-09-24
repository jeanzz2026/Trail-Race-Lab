from flask import Flask, render_template, request, jsonify
import os
from scraper import scrape_itra_results
from urllib.parse import urlparse

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or "trail-running-secret"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    url = request.form.get('url')
    
    # Basic URL validation
    if not url:
        return jsonify({'error': 'Please provide a URL'}), 400
    
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return jsonify({'error': 'Invalid URL format'}), 400
        
        # Perform scraping
        results = scrape_itra_results(url)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
