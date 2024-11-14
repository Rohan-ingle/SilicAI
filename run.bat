@echo off
start /min cmd /k "npm run dev"

start /min cmd /k "uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
start /min cmd /k "ngrok http --url=distinctly-thorough-pelican.ngrok-free.app 8000 "

start http://localhost:5173