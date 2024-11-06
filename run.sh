#!/bin/bash
source venv/bin/activate
while true; do
./mine.py mainnet;
echo "waiting 10 sec";
sleep 10;
done
