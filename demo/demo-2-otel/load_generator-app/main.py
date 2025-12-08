from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import threading
import time
import psutil
import logging

app = FastAPI()

# ---------------------------------------------------
# SAFE logger (DO NOT use uvicorn.access)
# ---------------------------------------------------
log = logging.getLogger("app")
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
log.addHandler(handler)

cpu_thread = None
ram_thread = None

_state = {
    "cpu_target": 0,
    "ram_target": 0,
    "cpu_running": False,
    "ram_running": False
}

class StartRequest(BaseModel):
    cpu_percent: int = 70
    ram_percent: int = 80

def cpu_load_loop(target_percent: int):
    _state["cpu_running"] = True
    period = 0.1
    busy = (target_percent / 100) * period
    sleep = max(period - busy, 0.001)

    while _state["cpu_running"]:
        start = time.time()
        while (time.time() - start) < busy:
            pass
        time.sleep(sleep)

def ram_load_loop(target_percent: int):
    _state["ram_running"] = True
    allocated = bytearray()
    step = 10 * 1024 * 1024
    check_interval = 0.5

    while _state["ram_running"]:
        vm = psutil.virtual_memory()
        current = vm.percent

        if current < target_percent - 1:
            try:
                allocated.extend(b'\0' * step)
            except MemoryError:
                allocated = allocated[:-step]
                time.sleep(1)

        elif current > target_percent + 1:
            if len(allocated) >= step:
                allocated = allocated[:-step]

        time.sleep(check_interval)

    del allocated

@app.post("/start")
def start(req: StartRequest):
    stop_internal()

    cpu_target = min(max(req.cpu_percent, 0), 100)
    ram_target = min(max(req.ram_percent, 0), 100)

    _state["cpu_target"] = cpu_target
    _state["ram_target"] = ram_target

    global cpu_thread, ram_thread

    if cpu_target > 0:
        cpu_thread = threading.Thread(target=cpu_load_loop, args=(cpu_target,), daemon=True)
        cpu_thread.start()

    if ram_target > 0:
        ram_thread = threading.Thread(target=ram_load_loop, args=(ram_target,), daemon=True)
        ram_thread.start()

    log.info(f"Started load → CPU={cpu_target}% RAM={ram_target}%")
    return {"status": "started"}

@app.post("/stop")
def stop():
    stop_internal()
    log.info("Stopped load")
    return {"status": "stopped"}

def stop_internal():
    _state["cpu_running"] = False
    _state["ram_running"] = False

@app.get("/status")
def status():
    vm = psutil.virtual_memory()
    return {
        "cpu_target": _state["cpu_target"],
        "ram_target": _state["ram_target"],
        "cpu_running": _state["cpu_running"],
        "ram_running": _state["ram_running"],
        "mem_percent": vm.percent,
        "mem_total_mb": int(vm.total / 1024 / 1024),
    }
