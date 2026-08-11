#!/bin/bash
# Waits for resume_webqsp to start, then to finish, then runs generate_split
# and verifies. Designed to run detached on the instance.
LOG=~/finalize.log
exec > >(tee "$LOG") 2>&1

echo "=== waiting for resume to start ($(date -u)) ==="
until [ -f ~/resume.log ] && grep -q "Encoding graphs" ~/resume.log; do sleep 5; done
echo "=== resume started, encoding graphs ($(date -u)) ==="

# wait for the python process to finish
while pgrep -f "resume_webqsp" >/dev/null; do sleep 60; done
echo "=== resume process ended ($(date -u)) ==="

N=$(ls ~/G-Retriever/dataset/webqsp/graphs/*.pt 2>/dev/null | wc -l)
echo "graph files on disk: $N"
echo "--- resume.log tail ---"
tail -6 ~/resume.log

if [ "$N" -ge 4699 ]; then
    echo "=== running generate_split ($(date -u)) ==="
    cd ~/G-Retriever
    ~/miniconda3/envs/gretriever/bin/python -c "from src.dataset.preprocess.webqsp import generate_split; generate_split()"
    echo "--- split dir ---"
    ls -la ~/G-Retriever/dataset/webqsp/split/
else
    echo "INCOMPLETE: expected >=4699 graphs, got $N. Skipping generate_split."
fi

echo "=== disk ==="
df -h /
echo "=== ALL DONE ($(date -u)) ==="
