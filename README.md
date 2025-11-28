가상환경 활성화 및 패키지 설치: pip install -r requirements.txt (아래에 requirements 파일도 포함).

서버 실행: uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload

브라우저에서 http://127.0.0.1:8000 접속.

https://drive.google.com/file/d/1ZQOWTOz-1pefqM7cg5XMZV2MYgxdRUju/view?usp=drive_link

폴더구조
focus_realtime/
 ├─ src/
 │   ├─ camera_worker.py
 │   ├─ model_utils.py
 │   ├─ pose_utils.py
 │   ├─ web_server.py
 │   └─ ...
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
