#!/usr/bin/env python3
"""
Beautiful Dashboard for Distributed ML Platform
Run this to see real-time cluster status
"""
from flask import Flask, render_template, jsonify, Response
import grpc
import json
import time
import threading
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protos import platform_pb2
from protos import platform_pb2_grpc

app = Flask(__name__)

# Global cache
class ClusterCache:
    def __init__(self):
        self.workers = []
        self.jobs = []
        self.stats = {
            'total_workers': 0,
            'active_jobs': 0,
            'total_gpus': 0,
            'total_cpu_cores': 0,
            'total_memory': 0
        }
        self.last_update = 0
        self.coordinator_host = 'localhost'
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
      
    
    def _update_loop(self):
        """Periodically fetch updates"""
        while True:
            try:
                self._fetch_status()
            except Exception as e:
                print(f"Error fetching status: {e}")
            time.sleep(2)
    
    def _fetch_status(self):
        """Fetch status from coordinator"""
        try:
            channel = grpc.insecure_channel(f"{self.coordinator_host}:50051")
            stub = platform_pb2_grpc.CoordinatorStub(channel)
            
            response = stub.GetStatus(platform_pb2.Empty())
            
            # Process workers
            self.workers = []
            for w in response.workers:
                self.workers.append({
                    'id': w.worker_id[:8] + '...',
                    'hostname': w.hostname,
                    'gpus': w.gpu_count,
                    'memory': round(w.memory_gb, 1),
                    'cpu_cores': w.cpu_cores
                })
            
            # Process jobs
            self.jobs = []
            active_jobs = 0
            for j in response.jobs:
                job_dict = {
                    'id': j.job_id,
                    'name': j.name,
                    'status': j.status,
                    'progress': round(j.progress * 100, 1),
                    'model': j.model_type,
                    'workers': j.num_workers,
                    'epoch': j.current_epoch,
                    'total_epochs': j.epochs,
                    'loss': round(j.current_loss, 4) if j.current_loss > 0 else 0,
                    'accuracy': round(j.current_accuracy * 100, 2) if j.current_accuracy > 0 else 0
                }
                
                if j.status == 'running':
                    active_jobs += 1
                
                self.jobs.append(job_dict)
            
            # Update stats
            self.stats = {
                'total_workers': len(self.workers),
                'active_jobs': active_jobs,
                'total_gpus': response.total_gpus,
                'total_cpu_cores': response.total_cpu_cores,
                'total_memory': round(response.total_memory_gb, 1),
                'last_update': datetime.now().strftime('%H:%M:%S')
            }
            
            self.last_update = time.time()
            
        except grpc.RpcError:
            self.stats['coordinator_status'] = 'offline'
        except Exception as e:
            print(f"Status fetch error: {e}")
    
    def get_data(self):
        return {
            'workers': self.workers,
            'jobs': self.jobs,
            'stats': self.stats
        }

cache = ClusterCache()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    """JSON API for real-time updates"""
    return jsonify(cache.get_data())

@app.route('/api/job/<job_id>')
def job_details(job_id):
    """Get specific job details"""
    try:
        channel = grpc.insecure_channel(f"{cache.coordinator_host}:50051")
        stub = platform_pb2_grpc.CoordinatorStub(channel)
        
        # Get all jobs and filter
        response = stub.GetStatus(platform_pb2.Empty())
        
        for job in response.jobs:
            if job.job_id == job_id:
                return jsonify({
                    'id': job.job_id,
                    'name': job.name,
                    'status': job.status,
                    'progress': job.progress,
                    'model': job.model_type,
                    'workers': job.num_workers,
                    'epochs': job.epochs,
                    'current_epoch': job.current_epoch,
                    'loss': job.current_loss,
                    'accuracy': job.current_accuracy,
                    'start_time': job.start_time
                })
        
        return jsonify({'error': 'Job not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)