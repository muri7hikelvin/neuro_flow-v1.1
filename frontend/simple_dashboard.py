
from flask import Flask
import socket

app = Flask(__name__)

@app.route('/')
def index():
    # Get IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
    except:
        ip = "localhost"
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Distributed ML Platform</title>
        <style>
            body {{ font-family: Arial; margin: 40px; background: #f0f2f5; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .card {{ background: white; padding: 30px; border-radius: 15px; 
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 20px 0; }}
            .header {{ background: linear-gradient(135deg, #667eea, #764ba2); 
                      color: white; padding: 20px; border-radius: 10px; }}
            .green {{ color: #00cc00; }}
            code {{ background: #f4f4f4; padding: 10px; border-radius: 5px; 
                    display: block; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Distributed ML Platform</h1>
                <p>Your cluster is running!</p>
            </div>
            
            <div class="card">
                <h2>✅ System Status</h2>
                <p><b>Coordinator:</b> <span class="green">● Online</span></p>
                <p><b>IP Address:</b> {ip}</p>
                <p><b>Port:</b> 50051</p>
                <p><b>Dashboard:</b> http://{ip}:5000</p>
            </div>
            
            <div class="card">
                <h2>📋 Quick Start</h2>
                <p><b>1. Start a worker (on another computer):</b></p>
                <code>python backend/worker.py --coordinator {ip}</code>
                
                <p><b>2. Submit a test job:</b></p>
                <code>python scripts/submit_demo_job.py --coordinator {ip}</code>
                
                <p><b>3. View in browser:</b></p>
                <code>open http://{ip}:5000</code>
            </div>
            
            <div class="card">
                <h2>📊 Real-time Stats</h2>
                <p>This dashboard will update every 2 seconds...</p>
                <div id="stats"></div>
            </div>
        </div>
        
        <script>
            function updateStats() {{
                fetch('/api/status')
                    .then(r => r.json())
                    .then(data => {{
                        let html = '<p>Workers: ' + (data.workers?.length || 0) + '</p>';
                        html += '<p>Jobs: ' + (data.jobs?.length || 0) + '</p>';
                        document.getElementById('stats').innerHTML = html;
                    }})
                    .catch(() => {{}});
            }}
            setInterval(updateStats, 2000);
            updateStats();
        </script>
    </body>
    </html>
    """

@app.route('/api/status')
def status():
    return {"workers": [], "jobs": []}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
