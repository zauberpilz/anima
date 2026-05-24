@echo off
wt -d "C:\Users\admin\Documents\Besseres LLM" wsl -d Ubuntu-24.04 -- bash -l -c "tmux -S /tmp/tmux-anima.sock attach -t anima"
