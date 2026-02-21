#!/bin/bash
# One-command cluster startup for demo

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}🚀 Starting Distributed ML Cluster${NC}"
echo -e "${BLUE}========================================${NC}"

# Generate protos
echo -e "\n📦 Generating gRPC code..."
./scripts/generate_protos.sh

# Start coordinator in background
echo -e "\n🔥 Starting coordinator..."
python backend/coordinator.py > logs/coordinator.log 2>&1 &
COORD_PID=$!
echo "   PID: $COORD_PID"

# Wait for coordinator to start
sleep 3

# Start dashboard
echo -e "\n📊 Starting dashboard..."
cd frontend
python app.py > ../logs/dashboard.log 2>&1 &
DASH_PID=$!
cd ..
echo "   PID: $DASH_PID"

echo -e "\n${GREEN}✅ Cluster started!${NC}"
echo -e "   Coordinator: localhost:50051"
echo -e "   Dashboard:   http://localhost:5000"
echo -e "\n📝 Logs are in ./logs/"
echo -e "\n${BLUE}To stop:${NC} kill $COORD_PID $DASH_PID"
echo ""