#!/bin/bash
# Kill the Question Predictor API server

echo "Stopping Question Predictor API..."

# Kill by port
fuser -k 5002/tcp 2>/dev/null

# Also kill by process name
pkill -f "python api.py" 2>/dev/null

sleep 1

# Verify
if fuser 5002/tcp 2>/dev/null; then
    echo "Warning: Server may still be running"
else
    echo "Server stopped successfully"
fi
