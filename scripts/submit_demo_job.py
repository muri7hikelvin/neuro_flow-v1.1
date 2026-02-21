#!/usr/bin/env python3
"""
Submit a simple MNIST training job to test the cluster
"""

import grpc
import sys
import os
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protos import platform_pb2
from protos import platform_pb2_grpc

def submit_job(coordinator_host='192.168.100.4'):
    print("\n" + "="*50)
    print("📤 SUBMITTING TRAINING JOB")
    print("="*50)
    
    # Connect to coordinator
    channel = grpc.insecure_channel(f"{coordinator_host}:50051")
    stub = platform_pb2_grpc.CoordinatorStub(channel)
    
    # Check status first
    status = stub.GetStatus(platform_pb2.Empty())
    print(f"\n📊 Cluster Status:")
    print(f"   Workers: {len(status.workers)}")
    print(f"   Total CPUs: {status.total_cpu_cores}")
    print(f"   Total RAM: {status.total_memory_gb:.1f}GB")
    
    # Submit job
    print("\n🚀 Submitting MNIST training job...")
    request = platform_pb2.JobRequest(
        model_type='mnist_cnn',
        num_workers=1,  # Use just 1 worker for now
        epochs=2,       # Just 2 epochs for quick demo
        batch_size=64,
        dataset='mnist',
        job_name=f"MNIST-Demo-{datetime.now().strftime('%H:%M:%S')}"
    )
    
    response = stub.SubmitJob(request)
    
    if response.accepted:
        print(f"\n✅ Job Accepted!")
        print(f"   Job ID: {response.job_id}")
        print(f"   Message: {response.message}")
        print(f"\n📊 Watch progress:")
        print(f"   Dashboard: http://{coordinator_host}:5001")
        print(f"   Worker terminal: Look for training output")
        
        # Monitor for a bit
        print("\n📈 Monitoring for 30 seconds...")
        for i in range(30):
            status = stub.GetStatus(platform_pb2.Empty())
            running_jobs = [j for j in status.jobs if j.status == 'running']
            if running_jobs:
                job = running_jobs[0]
                print(f"   Epoch: {job.current_epoch}/{job.epochs}, Loss: {job.current_loss:.4f}, Acc: {job.current_accuracy*100:.2f}%", end='\r')
            else:
                print("   Waiting for job to start...", end='\r')
            time.sleep(2)
        print("\n\n✅ Monitoring complete! Check dashboard for final results.")
        
    else:
        print(f"\n❌ Job rejected: {response.message}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinator', default='192.168.100.4', help='Coordinator IP')
    args = parser.parse_args()
    
    submit_job(args.coordinator)