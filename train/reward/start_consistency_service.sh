#!/bin/bash
# This script is used to start the Swift-based consistency verification model service on GPU 4

# Set environment variables to use GPU 4
export CUDA_VISIBLE_DEVICES=4

# Check if the service is already running
if pgrep -f "start_consistency_model_server.py" > /dev/null; then
    echo "Consistency verification model service is already running"
    exit 0
fi

# Ensure the log directory exists
mkdir -p train/logs

# Start the model service
echo "Starting Swift-based consistency verification model service on GPU 4..."
nohup python train/start_consistency_model_server.py > train/consistency_service_swift.log 2>&1 &

# Get the process ID
PID=$!
echo "Service started, process ID: $PID"

# Wait for the service to start
echo "Waiting for service to start..."
for i in {1..30}; do
    echo -n "."
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "\nSwift-based consistency verification model service started successfully"
        exit 0
    fi
    sleep 2
done

echo -e "\nService startup timed out, check the log file..."
tail -n 50 train/consistency_service_swift.log

# If the service fails to start, try the backup model
echo "Service startup failed, please check the log file: train/consistency_service_swift.log"
exit 1
