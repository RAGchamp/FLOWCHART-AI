@echo off
rem Starts the FlowTrace service used by the Flow Chart App's "Trace it" button.
rem Keep this window open while you trace; close it (or press Ctrl+C) to stop.
title FlowTrace
cd /d "%~dp0service"
python flowtrace_server.py
if errorlevel 1 pause
