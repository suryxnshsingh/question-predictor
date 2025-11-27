#!/bin/bash
# Start the Question Predictor API server

cd "$(dirname "$0")"

# Kill any existing server on port 5002
fuser -k 5002/tcp 2>/dev/null
sleep 1

# Activate virtual environment
source venv/bin/activate

# Start server
echo "Starting Question Predictor API on port 5002..."
nohup python api.py > logs/server.log 2>&1 &

# Wait and check
sleep 3
if curl -s http://localhost:5002/health > /dev/null; then
    echo "Server started successfully!"
    echo "Swagger UI: http://localhost:5002/docs"
    echo "Logs: logs/server.log"
else
    echo "Failed to start server. Check logs/server.log"
    exit 1
fi
