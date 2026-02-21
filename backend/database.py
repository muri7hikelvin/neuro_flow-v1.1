"""
SQLite database for job persistence and tracking
"""
import sqlite3
import json
import time
from datetime import datetime
from typing import Dict, List, Optional
import threading

class JobDatabase:
    def __init__(self, db_path='jobs.db'):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.init_db()
    
    def init_db(self):
        """Create tables if they don't exist"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Jobs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    name TEXT,
                    model_type TEXT,
                    dataset TEXT,
                    num_workers INTEGER,
                    epochs INTEGER,
                    batch_size INTEGER,
                    status TEXT,
                    progress REAL,
                    current_loss REAL,
                    current_accuracy REAL,
                    current_epoch INTEGER,
                    start_time INTEGER,
                    end_time INTEGER,
                    error_message TEXT,
                    config_json TEXT
                )
            ''')
            
            # Job logs table (for streaming)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS job_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    step INTEGER,
                    epoch INTEGER,
                    loss REAL,
                    accuracy REAL,
                    timestamp INTEGER,
                    worker_id TEXT,
                    rank INTEGER,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                )
            ''')
            
            # Workers table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    hostname TEXT,
                    ip_address TEXT,
                    port INTEGER,
                    gpu_count INTEGER,
                    memory_gb REAL,
                    cpu_cores INTEGER,
                    status TEXT,
                    last_heartbeat INTEGER,
                    current_job TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
    
    def create_job(self, job_id: str, job_config: Dict) -> bool:
        """Create a new job entry"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO jobs (
                        job_id, name, model_type, dataset, num_workers, 
                        epochs, batch_size, status, progress, 
                        current_loss, current_accuracy, current_epoch,
                        start_time, config_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    job_id,
                    job_config.get('name', job_id),
                    job_config['model_type'],
                    job_config['dataset'],
                    job_config['num_workers'],
                    job_config['epochs'],
                    job_config['batch_size'],
                    'pending',
                    0.0,
                    0.0,
                    0.0,
                    0,
                    int(time.time()),
                    json.dumps(job_config)
                ))
                
                conn.commit()
                conn.close()
                return True
            except Exception as e:
                print(f"Error creating job: {e}")
                return False
    
    def update_job_status(self, job_id: str, status: str, progress: float = None,
                          loss: float = None, accuracy: float = None,
                          epoch: int = None):
        """Update job status and metrics"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                updates = []
                params = []
                
                if status:
                    updates.append("status = ?")
                    params.append(status)
                    
                    if status in ['completed', 'failed']:
                        updates.append("end_time = ?")
                        params.append(int(time.time()))
                
                if progress is not None:
                    updates.append("progress = ?")
                    params.append(progress)
                
                if loss is not None:
                    updates.append("current_loss = ?")
                    params.append(loss)
                
                if accuracy is not None:
                    updates.append("current_accuracy = ?")
                    params.append(accuracy)
                
                if epoch is not None:
                    updates.append("current_epoch = ?")
                    params.append(epoch)
                
                if updates:
                    query = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?"
                    params.append(job_id)
                    cursor.execute(query, params)
                    conn.commit()
                
                conn.close()
            except Exception as e:
                print(f"Error updating job: {e}")
    
    def add_log_entry(self, job_id: str, log_entry: Dict):
        """Add a training log entry"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO job_logs (
                        job_id, step, epoch, loss, accuracy, 
                        timestamp, worker_id, rank
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    job_id,
                    log_entry.get('step', 0),
                    log_entry.get('epoch', 0),
                    log_entry.get('loss', 0.0),
                    log_entry.get('accuracy', 0.0),
                    log_entry.get('timestamp', int(time.time() * 1000)),
                    log_entry.get('worker_id', ''),
                    log_entry.get('rank', 0)
                ))
                
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Error adding log: {e}")
    
    def register_worker(self, worker_info: Dict) -> bool:
        """Register or update worker"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO workers (
                        worker_id, hostname, ip_address, port, gpu_count,
                        memory_gb, cpu_cores, status, last_heartbeat, current_job
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    worker_info['worker_id'],
                    worker_info['hostname'],
                    worker_info.get('ip_address', ''),
                    worker_info['port'],
                    worker_info['gpu_count'],
                    worker_info['memory_gb'],
                    worker_info['cpu_cores'],
                    'online',
                    int(time.time()),
                    worker_info.get('current_job', '')
                ))
                
                conn.commit()
                conn.close()
                return True
            except Exception as e:
                print(f"Error registering worker: {e}")
                return False
    
    def update_worker_heartbeat(self, worker_id: str, metrics: Dict):
        """Update worker heartbeat"""
        with self.lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE workers 
                    SET last_heartbeat = ?, 
                        cpu_utilization = ?,
                        memory_utilization = ?,
                        gpu_utilization = ?
                    WHERE worker_id = ?
                ''', (
                    int(time.time()),
                    metrics.get('cpu_util', 0),
                    metrics.get('mem_util', 0),
                    metrics.get('gpu_util', 0),
                    worker_id
                ))
                
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Error updating heartbeat: {e}")
    
    def get_all_jobs(self) -> List[Dict]:
        """Get all jobs"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM jobs ORDER BY start_time DESC
            ''')
            
            jobs = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return jobs
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        """Get specific job"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM jobs WHERE job_id = ?', (job_id,))
            row = cursor.fetchone()
            conn.close()
            
            return dict(row) if row else None
    
    def get_job_logs(self, job_id: str, limit: int = 100) -> List[Dict]:
        """Get job training logs"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM job_logs 
                WHERE job_id = ? 
                ORDER BY step DESC 
                LIMIT ?
            ''', (job_id, limit))
            
            logs = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return logs
    
    def get_all_workers(self) -> List[Dict]:
        """Get all workers"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Clean up stale workers (no heartbeat for >30 seconds)
            stale_time = int(time.time()) - 30
            cursor.execute('''
                UPDATE workers 
                SET status = 'offline' 
                WHERE last_heartbeat < ?
            ''', (stale_time,))
            conn.commit()
            
            cursor.execute('SELECT * FROM workers ORDER BY last_heartbeat DESC')
            workers = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return workers