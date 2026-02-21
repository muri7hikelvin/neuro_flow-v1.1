#!/bin/bash
# Generate gRPC Python code from proto files

cd "$(dirname "$0")/.."

echo "📦 Generating gRPC code from protos/platform.proto..."

python -m grpc_tools.protoc \
    -I./protos \
    --python_out=./protos \
    --grpc_python_out=./protos \
    ./protos/platform.proto

# Fix imports
sed -i 's/import platform_pb2/from . import platform_pb2/' ./protos/platform_pb2_grpc.py

echo "✅ gRPC code generated successfully!"