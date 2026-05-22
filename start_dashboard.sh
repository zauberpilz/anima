#!/bin/bash
# Startet das Anima Dashboard im Hintergrund
# Autorun: (crontab -l 2>/dev/null; echo "@reboot /home/anima/start_dashboard.sh") | crontab -

cd /home/anima/src
source /home/anima/venv/bin/activate
nohup python3 dashboard.py > /home/anima/dashboard.log 2>&1 &
echo "Dashboard PID: $!"
