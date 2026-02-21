"""
Coordinator Server - Central brain of the distributed system
Run this on your main computer
"""
import grpc
from concurrent import futures
import threading
import time
import socket
import uuid
import json
import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List
import torch
# Add these imports
import threading
import subprocess
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protos import platform_pb2
from protos import platform_pb2_grpc
from backend.database import JobDatabase

class CoordinatorServicer(platform_pb2_grpc.CoordinatorServicer):
    def __init__(self, host='0.0.0.0', port=50051):
        self.host = host
        self.port = port
        self.db = JobDatabase()
        
        # In-memory state for fast access
        self.workers: Dict[str, Dict] = {}
        self.jobs: Dict[str, Dict] = {}
        self.active_training: Dict[str, List[threading.Thread]] = {}
        
        # Get local IP
        self.local_ip = self._get_local_ip()
        
        # Start heartbeat monitor
        self.start_heartbeat_monitor()
        
        print(f"\n{'='*60}")
        print(f"🚀 DISTRIBUTED ML COORDINATOR")
        print(f"{'='*60}")
        print(f"Hostname: {socket.gethostname()}")
        print(f"IP Address: {self.local_ip}")
        print(f"Port: {port}")
        print(f"Dashboard: http://{self.local_ip}:5000")
        print(f"{'='*60}\n")
    
    def _get_local_ip(self):
        """Get local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return '127.0.0.1'
    
    def start_heartbeat_monitor(self):
        """Monitor worker heartbeats and mark stale workers"""
        def monitor():
            while True:
                current_time = time.time()
                stale_workers = []
                
                for worker_id, worker in self.workers.items():
                    if current_time - worker.get('last_heartbeat', 0) > 30:
                        stale_workers.append(worker_id)
                
                for worker_id in stale_workers:
                    print(f"⚠️ Worker {worker_id} timed out")
                    self.workers[worker_id]['status'] = 'offline'
                    # Update DB
                    self.db.update_worker_heartbeat(worker_id, {})
                
                time.sleep(10)
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def RegisterWorker(self, request, context):
        """Worker joins the cluster"""
        worker_id = request.worker_id
        
        self.workers[worker_id] = {
            'worker_id': worker_id,
            'hostname': request.hostname,
            'ip_address': request.ip_address,
            'port': request.port,
            'gpu_count': request.gpu_count,
            'memory_gb': request.memory_gb,
            'cpu_cores': request.cpu_cores,
            'status': 'online',
            'last_heartbeat': time.time(),
            'current_job': None
        }
        
        # Store in database
        self.db.register_worker(self.workers[worker_id])
        
        print(f"\n✅ Worker registered: {request.hostname}")
        print(f"   ID: {worker_id[:8]}...")
        print(f"   GPUs: {request.gpu_count}, RAM: {request.memory_gb:.1f}GB")
        print(f"   IP: {request.ip_address}")
        
        return platform_pb2.RegistrationResponse(
            success=True,
            message=f"Welcome to the cluster!",
            assigned_id=worker_id
        )
    
    def Heartbeat(self, request, context):
        """Worker sends periodic heartbeat"""
        worker_id = request.worker_id
        
        if worker_id in self.workers:
            self.workers[worker_id]['last_heartbeat'] = time.time()
            self.workers[worker_id]['cpu_util'] = request.cpu_utilization
            self.workers[worker_id]['mem_util'] = request.memory_utilization
            self.workers[worker_id]['gpu_util'] = request.gpu_utilization
            
            # Update DB
            self.db.update_worker_heartbeat(worker_id, {
                'cpu_util': request.cpu_utilization,
                'mem_util': request.memory_utilization,
                'gpu_util': request.gpu_utilization
            })
            
            return platform_pb2.HeartbeatResponse(acknowledged=True)
        
        return platform_pb2.HeartbeatResponse(acknowledged=False)
    
    def SubmitJob(self, request, context):
        """User submits a training job"""
        print(f"\n📥 Job submission received:")
        print(f"   Model: {request.model_type}")
        print(f"   Workers: {request.num_workers}")
        print(f"   Dataset: {request.dataset}")
        
        # Check available workers
        available_workers = [
            w for w in self.workers.values() 
            if w['status'] == 'online' and w.get('current_job') is None
        ]
        
        if len(available_workers) < request.num_workers:
            msg = f"Need {request.num_workers} workers, only {len(available_workers)} available"
            print(f"   ❌ {msg}")
            return platform_pb2.JobResponse(
                accepted=False,
                job_id="",
                message=msg
            )
        
        # Create job ID
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        # Job config
        job_config = {
            'name': request.job_name or f"Training-{request.model_type}",
            'job_id': job_id,
            'model_type': request.model_type,
            'dataset': request.dataset,
            'num_workers': request.num_workers,
            'epochs': request.epochs,
            'batch_size': request.batch_size,
            'status': 'pending',
            'created_at': time.time()
        }
        
        # Store in database
        self.db.create_job(job_id, job_config)
        
        # Start training in background
        selected_workers = available_workers[:request.num_workers]
        thread = threading.Thread(
            target=self._run_distributed_training,
            args=(job_id, job_config, selected_workers)
        )
        thread.start()
        self.active_training[job_id] = [thread]
        
        # Estimate time (simplified)
        est_time = request.epochs * 10  # Rough estimate
        
        print(f"   ✅ Job accepted! ID: {job_id}")
        print(f"   Workers assigned: {[w['hostname'] for w in selected_workers]}")
        
        return platform_pb2.JobResponse(
            accepted=True,
            job_id=job_id,
            message=f"Job started with {request.num_workers} workers",
            estimated_time_seconds=est_time
        )
    
    def _run_distributed_training(self, job_id, job_config, workers):
        """Orchestrate distributed training including coordinator"""
        print(f"\n🚀 Starting job {job_id} on {len(workers)} workers")
        
        # Mark workers as busy
        for worker in workers:
            worker['current_job'] = job_id
        
        self.db.update_job_status(job_id, 'running')
        
        # Set up master address (use coordinator IP)
        master_addr = self.local_ip
        master_port = 23456
        
        # Include coordinator as worker 0 if it has resources
        all_workers = workers.copy()
        
        # Start each worker
        threads = []
        for rank, worker in enumerate(all_workers):
            thread = threading.Thread(
                target=self._start_worker_training,
                args=(job_id, worker, rank, len(all_workers), master_addr, master_port, job_config)
            )
            thread.start()
            threads.append(thread)
            print(f"   ▶️ Worker {rank} ({worker['hostname']}) started")
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # Mark job as complete
        self.db.update_job_status(job_id, 'completed', progress=1.0)
        
        # Free workers
        for worker in workers:
            worker['current_job'] = None
        
        print(f"\n✅ Job {job_id} completed successfully!")
    
    def _start_worker_training(self, job_id, worker, rank, world_size, master_addr, master_port, job_config):
        """Connect to worker and start training task"""
        try:
            # Connect to worker
            channel = grpc.insecure_channel(f"{worker['ip_address']}:{worker['port']}")
            stub = platform_pb2_grpc.WorkerStub(channel)
            
            # Create training task
            task = platform_pb2.TrainingTask(
                job_id=job_id,
                model_type=job_config['model_type'],
                world_size=world_size,
                rank=rank,
                master_addr=master_addr,
                master_port=master_port,
                epochs=job_config['epochs'],
                batch_size=job_config['batch_size'],
                dataset=job_config['dataset'],
                total_steps=job_config['epochs'] * 100  # Rough estimate
            )
            
            # Start training (this streams updates)
            updates = stub.StartTraining(task)
            
            # Process updates
            for update in updates:
                # Store in database
                log_entry = {
                    'step': update.step,
                    'epoch': update.epoch,
                    'loss': update.loss,
                    'accuracy': update.accuracy,
                    'timestamp': int(time.time() * 1000),
                    'worker_id': worker['worker_id'],
                    'rank': rank
                }
                self.db.add_log_entry(job_id, log_entry)
                
                # Update job summary (only from rank 0)
                if rank == 0:
                    progress = (update.epoch * 100 + update.step) / (job_config['epochs'] * 100)
                    self.db.update_job_status(
                        job_id, 
                        'running',
                        progress=min(progress, 0.99),
                        loss=update.loss,
                        accuracy=update.accuracy,
                        epoch=update.epoch
                    )
            
        except Exception as e:
            print(f"❌ Worker {rank} for job {job_id} failed: {e}")
            self.db.update_job_status(job_id, 'failed', error_message=str(e))
    
    def GetStatus(self, request, context):
        """Return current cluster status"""
        # Get all workers
        worker_infos = []
        total_cpu = 0
        total_memory = 0
        total_gpus = 0
        
        for worker_id, worker in self.workers.items():
            if worker.get('status') == 'online':
                worker_infos.append(platform_pb2.WorkerInfo(
                    worker_id=worker_id,
                    hostname=worker['hostname'],
                    port=worker['port'],
                    gpu_count=worker['gpu_count'],
                    memory_gb=worker['memory_gb'],
                    cpu_cores=worker['cpu_cores'],
                    ip_address=worker['ip_address']
                ))
                total_cpu += worker['cpu_cores']
                total_memory += worker['memory_gb']
                total_gpus += worker['gpu_count']
        
        # Get jobs from database
        jobs = self.db.get_all_jobs()
        job_summaries = []
        
        for job in jobs[:10]:  # Last 10 jobs
            job_summaries.append(platform_pb2.JobSummary(
                job_id=job['job_id'],
                name=job['name'],
                status=job['status'],
                progress=job['progress'],
                num_workers=job['num_workers'],
                model_type=job['model_type'],
                epochs=job['epochs'],
                current_epoch=job['current_epoch'],
                current_loss=job['current_loss'],
                current_accuracy=job['current_accuracy'],
                start_time=job['start_time'],
                end_time=job.get('end_time', 0)
            ))
        
        return platform_pb2.ClusterStatus(
            workers=worker_infos,
            jobs=job_summaries,
            total_cpu_cores=total_cpu,
            total_memory_gb=total_memory,
            total_gpus=total_gpus
        )
    
    def StreamJobLogs(self, request, context):
        """Stream job logs for real-time monitoring"""
        job_id = request.job_id
        last_step = request.since_step
        
        while True:
            logs = self.db.get_job_logs(job_id, limit=10)
            for log in logs:
                if log['step'] > last_step:
                    yield platform_pb2.LogEntry(
                        step=log['step'],
                        epoch=log['epoch'],
                        loss=log['loss'],
                        accuracy=log['accuracy'],
                        timestamp=log['timestamp'],
                        worker_id=log['worker_id'],
                        rank=log['rank']
                    )
                    last_step = log['step']
            
            time.sleep(2)  # Poll every 2 seconds




    def start_self_as_worker(self):
        """Start the coordinator itself as worker 0"""
        print("\n🖥️ Starting coordinator as Worker 0...")
        
        # Register self as a worker
        worker_id = f"worker_coordinator_{uuid.uuid4().hex[:6]}"
        
        self.workers[worker_id] = {
            'worker_id': worker_id,
            'hostname': socket.gethostname(),
            'ip_address': self.local_ip,
            'port': 50053,  # Different port for self-worker
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'memory_gb': self._get_system_memory(),
            'cpu_cores': os.cpu_count(),
            'status': 'online',
            'last_heartbeat': time.time(),
            'current_job': None,
            'is_coordinator': True
        }
        
        # Store in database
        self.db.register_worker(self.workers[worker_id])
        
        print(f"✅ Coordinator registered as worker:")
        print(f"   ID: {worker_id[:8]}...")
        print(f"   GPUs: {self.workers[worker_id]['gpu_count']}")
        print(f"   RAM: {self.workers[worker_id]['memory_gb']:.1f}GB")
        print(f"   CPU Cores: {self.workers[worker_id]['cpu_cores']}")
        
        # Start heartbeat thread for self
        self.start_self_heartbeat(worker_id)
        
        return worker_id

    def _get_system_memory(self):
        """Get system memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except:
            return 16  # Default if psutil not available

    def start_self_heartbeat(self, worker_id):
        """Send heartbeats for self-worker"""
        def heartbeat():
            while True:
                try:
                    # Update last heartbeat
                    self.workers[worker_id]['last_heartbeat'] = time.time()
                    
                    # Get system metrics
                    cpu_util = self._get_cpu_utilization()
                    mem_util = self._get_memory_utilization()
                    
                    self.workers[worker_id]['cpu_util'] = cpu_util
                    self.workers[worker_id]['mem_util'] = mem_util
                    
                    # Update DB
                    self.db.update_worker_heartbeat(worker_id, {
                        'cpu_util': cpu_util,
                        'mem_util': mem_util,
                        'gpu_util': 0
                    })
                    
                except Exception as e:
                    print(f"Self-heartbeat error: {e}")
                
                time.sleep(5)
        
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()

    def _get_cpu_utilization(self):
        """Get CPU utilization"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0

    def _get_memory_utilization(self):
        """Get memory utilization"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except:
            return 0       

def serve():
    """Start the coordinator server"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50051, help='Port to listen on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()
    
    # Create gRPC server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=20),
        maximum_concurrent_rpcs=100
    )
    
    # Add servicer
    servicer = CoordinatorServicer(args.host, args.port)
    platform_pb2_grpc.add_CoordinatorServicer_to_server(servicer, server)
    
    # Start server
    server.add_insecure_port(f'{args.host}:{args.port}')
    server.start()
    
    print(f"\n🔥 Coordinator gRPC server running on port {args.port}")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down coordinator...")
        server.stop(0)

if __name__ == '__main__':
    serve()