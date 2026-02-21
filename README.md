# Distributed ML Platform - Complete System

A production-ready distributed system for training machine learning models across multiple computers with a beautiful real-time dashboard.

## 🚀 Quick Start (3 Computers Demo)

### Prerequisites
- Python 3.8+ on all computers
- All computers on same network
- Firewall allows ports 50051, 50052, 5000

### Step 1: Setup on ALL Computers
```bash
# Clone or copy this folder to each computer
cd distributed-ml-platform

# Install dependencies
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# Generate gRPC code
./scripts/generate_protos.sh