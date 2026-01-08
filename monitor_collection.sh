#!/bin/bash
# Monitor data collection progress

LOG_FILE="collection_500.log"
PID=397174

echo "========================================"
echo "SALUS Data Collection Monitor"
echo "========================================"
echo "PID: $PID"
echo ""

# Check if process is running
if ps -p $PID > /dev/null; then
    echo "Status: ✅ RUNNING"
else
    echo "Status: ⚠️  STOPPED"
fi

echo ""
echo "Latest progress:"
echo "----------------------------------------"
tail -20 "$LOG_FILE" | grep -E "Collecting episodes|Progress|Success|Failure|Storage|Episode"

echo ""
echo "========================================"
echo "Commands:"
echo "  Watch live: tail -f $LOG_FILE"
echo "  Kill: kill $PID"
echo "  Check process: ps -p $PID"
echo "========================================"
