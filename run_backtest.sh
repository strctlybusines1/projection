#!/bin/bash
cd /sessions/youthful-funny-faraday/mnt/Code/projection
python3 mdn_projection.py --backtest 2>&1 | tee backtest_run.log
