#!/usr/bin/env python3
"""
Worker Node - Runs on each computer that helps train
"""
import grpc
from concurrent import futures
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import time
import socket
import uuid
import argparse
import threading
import psutil
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protos import platform_pb2
from protos import platform_pb2_grpc
from backend.models import create_model, count_parameters

class WorkerNode:
    def __init__(self, coordinator_host: str, coordinator_port: int = 50051):
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        self.worker_id = f"worker_{socket.gethostname()}_{uuid.uuid4().hex[:6]}"
        self.server_port = 50052
        self.running = True
        self.current_job = None
        
        # Get system info
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.cpu_cores = psutil.cpu_count()
        self.hostname = socket.gethostname()
        self.ip_address = self._get_local_ip()
        
        print(f"\n{'='*50}")
        print(f"🔧 WORKER NODE INITIALIZING")
        print(f"{'='*50}")
        print(f"Worker ID: {self.worker_id}")
        print(f"Hostname: {self.hostname}")
        print(f"IP: {self.ip_address}")
        print(f"GPUs: {self.gpu_count}")
        print(f"RAM: {self.memory_gb:.1f}GB")
        print(f"CPU Cores: {self.cpu_cores}")
        print(f"{'='*50}\n")
        
        # Register with coordinator
        self.register()
        
        # Start heartbeat thread
        self.start_heartbeat()
    
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
    
    def register(self):
        """Register with coordinator"""
        try:
            channel = grpc.insecure_channel(f"{self.coordinator_host}:{self.coordinator_port}")
            stub = platform_pb2_grpc.CoordinatorStub(channel)
            
            worker_info = platform_pb2.WorkerInfo(
                worker_id=self.worker_id,
                hostname=self.hostname,
                port=self.server_port,
                gpu_count=self.gpu_count,
                memory_gb=self.memory_gb,
                cpu_cores=self.cpu_cores,
                ip_address=self.ip_address
            )
            
            response = stub.RegisterWorker(worker_info)
            
            if response.success:
                print(f"✅ Successfully registered with coordinator at {self.coordinator_host}")
            else:
                print(f"❌ Registration failed: {response.message}")
                
        except Exception as e:
            print(f"❌ Failed to register: {e}")
            print(f"   Make sure coordinator is running at {self.coordinator_host}:{self.coordinator_port}")
    
    def start_heartbeat(self):
        """Send periodic heartbeats to coordinator"""
        def heartbeat_loop():
            while self.running:
                try:
                    channel = grpc.insecure_channel(f"{self.coordinator_host}:{self.coordinator_port}")
                    stub = platform_pb2_grpc.CoordinatorStub(channel)
                    
                    # Get current resource usage
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    mem_percent = psutil.virtual_memory().percent
                    
                    # GPU utilization (simplified)
                    gpu_util = 0
                    if torch.cuda.is_available():
                        try:
                            gpu_util = torch.cuda.utilization()
                        except:
                            pass
                    
                    heartbeat = platform_pb2.HeartbeatRequest(
                        worker_id=self.worker_id,
                        cpu_utilization=cpu_percent,
                        memory_utilization=mem_percent,
                        gpu_utilization=gpu_util,
                        active_jobs=1 if self.current_job else 0
                    )
                    
                    response = stub.Heartbeat(heartbeat)
                    
                    if response.acknowledged:
                        print(f"💓 Heartbeat sent - CPU: {cpu_percent}%, MEM: {mem_percent}%, GPU: {gpu_util}%", end='\r')
                    
                except Exception as e:
                    print(f"Heartbeat failed: {e}", end='\r')
                
                time.sleep(5)
        
        thread = threading.Thread(target=heartbeat_loop, daemon=True)
        thread.start()
        print("💓 Heartbeat thread started")
    
    def start_grpc_server(self):
        """Start gRPC server for receiving tasks"""
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        
        # Create and add servicer
        servicer = WorkerServicer(self)
        platform_pb2_grpc.add_WorkerServicer_to_server(servicer, server)
        
        server.add_insecure_port(f'[::]:{self.server_port}')
        server.start()
        
        print(f"🚀 Worker gRPC server listening on port {self.server_port}")
        print("   Waiting for training tasks...\n")
        
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            print("\n👋 Shutting down worker...")
            self.running = False
            server.stop(0)

class WorkerServicer(platform_pb2_grpc.WorkerServicer):
    def __init__(self, worker_node):
        self.worker = worker_node
        self.current_training = None
      
    
    def StartTraining(self, request, context):
        """Coordinator calls this to start training"""
        print(f"\n🎯 Starting training task:")
        print(f"   Job ID: {request.job_id}")
        print(f"   Model: {request.model_type}")
        print(f"   Rank: {request.rank}/{request.world_size-1}")
        print(f"   Dataset: {request.dataset}")
        
        self.worker.current_job = request.job_id
        
        # Set up distributed training
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{request.rank % self.worker.gpu_count}')
            torch.cuda.set_device(device)
            backend = 'nccl'
        else:
            device = torch.device('cpu')
            backend = 'gloo'
        
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method=f'tcp://{request.master_addr}:{request.master_port}',
            rank=request.rank,
            world_size=request.world_size
        )
        
        # Create model
        model = create_model(request.model_type, device)
        model = DDP(model, device_ids=[device.index] if torch.cuda.is_available() else None)
        
        print(f"   Model parameters: {count_parameters(model):,}")
        
        # Load dataset
        train_loader = self._create_dataloader(
            request.dataset,
            request.batch_size,
            request.world_size,
            request.rank
        )
        
        # Set up training
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        total_batches = len(train_loader)
        
        # Training loop
        model.train()
        step = 0
        
        for epoch in range(request.epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            start_time = time.time()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                pred = output.argmax(dim=1)
                epoch_correct += pred.eq(target).sum().item()
                epoch_total += target.size(0)
                
                # Calculate metrics
                avg_loss = epoch_loss / (batch_idx + 1)
                accuracy = 100.0 * epoch_correct / epoch_total if epoch_total > 0 else 0
                time_per_step = (time.time() - start_time) * 1000 / (batch_idx + 1)
                
                # Print progress
                if batch_idx % 10 == 0:
                    print(f"   Worker {request.rank}: Epoch {epoch}/{request.epochs-1}, "
                          f"Batch {batch_idx}/{total_batches-1}, "
                          f"Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%", end='\r')
                    
                    # Send update to coordinator
                    yield platform_pb2.TrainingUpdate(
                        epoch=epoch,
                        step=step,
                        loss=avg_loss,
                        accuracy=accuracy / 100.0,
                        samples_processed=epoch_total,
                        learning_rate=0.01,
                        time_per_step_ms=time_per_step
                    )
                    
                    step += 1
            
            # End of epoch
            epoch_accuracy = 100.0 * epoch_correct / epoch_total
            print(f"\n   Worker {request.rank}: Completed epoch {epoch} - "
                  f"Loss: {epoch_loss/total_batches:.4f}, Acc: {epoch_accuracy:.2f}%")
        
        # Clean up
        dist.destroy_process_group()
        self.worker.current_job = None
        print(f"\n✅ Worker {request.rank} completed job {request.job_id}")
    
    def _create_dataloader(self, dataset_name, batch_size, world_size, rank):
        """Create distributed dataloader"""
        # Data transforms
        if dataset_name == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        elif dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        # DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return loader
    
    def StopTraining(self, request, context):
        """Stop current training"""
        return platform_pb2.StopResponse(stopped=True)
    
    def Ping(self, request, context):
        """Health check"""
        return platform_pb2.Pong(alive=True, timestamp=time.time())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinator', required=True, help='Coordinator IP address')
    parser.add_argument('--port', type=int, default=50051, help='Coordinator port')
    args = parser.parse_args()
    
    # Create and start worker
    worker = WorkerNode(args.coordinator, args.port)
    worker.start_grpc_server()

if __name__ == '__main__':
    main()