폴더구조
focus_realtime/
 ├─ src/
 │   ├─ camera_worker.py
 │   ├─ model_utils.py
 │   ├─ pose_utils.py
 │   ├─ web_server.py
 │   ├─ log_utils.py
 │   ├─ __init__.py
 │   └─ main.py
 ├─ models/
 │   └─ best_model.pth
 ├─ static/
 │   └─ index.html
 ├─ screenshots/
 ├─ runs/
 ├─ meta.csv
 ├─ README.md
 ├─ requirements.txt
 └─ venv/"# focus_realtime" 

cd ~/focus_realtime
source venv/bin/activate
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

sudo yum install -y mesa-libGL

S3 폴더 비우기
ACM인증서
Route53

---
가비아 네임서버