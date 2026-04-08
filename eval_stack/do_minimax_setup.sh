#!/bin/bash
# Bootstrap script for DigitalOcean droplet running MiniMax benchmarks.
# Uploaded and executed by the orchestrator.
set -e

echo "=== Setting up MiniMax benchmark runner ==="

# Install Python and pip
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv > /dev/null 2>&1

# Create workspace
mkdir -p /eval/src /eval/data /eval/results

echo "=== Installing Python packages ==="
pip3 install -q openai datasets pandas python-dotenv tqdm aiohttp

echo "=== Setup complete ==="
