#!/bin/bash
source venv/bin/activate
pip install -r requirements.txt
while true; do
./mine.py mainnet;
echo "waiting 10 sec";
sleep 10;
done
