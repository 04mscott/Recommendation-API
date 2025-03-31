#!/usr/bin/env bash
# Update package lists
apt-get update
# Install FFmpeg
apt-get install -y ffmpeg
# Install dependencies
pip install -r requirements.txt
