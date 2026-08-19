#!/bin/bash
while pgrep -u zengjj -f 'multiset/run_scaling.py' >/dev/null 2>&1; do sleep 60; done
sleep 30
cd /curie-home/zengjj/Renormalizer/multiset_202607/2DHolsteinScaling
./calibrate_hop.sh
