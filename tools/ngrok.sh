#!/bin/bash

if [ ! -d "tmp" ]; then mkdir "tmp"; fi

wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip -P 'tmp'
unzip -o tmp/ngrok-stable-linux-amd64.zip -d tmp
rm tmp/ngrok-stable-linux-amd64.zip

tensorboard --logdir ${1:-tmp/out} --host 0.0.0.0 --port 6006 &

if [ ! -z $2 ]; then ./tmp/ngrok authtoken $2; fi
./tmp/ngrok http 6006 > /dev/null &

curl -s http://localhost:4040/api/tunnels | \
    python3 -c "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
