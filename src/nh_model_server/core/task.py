import threading
import time
from enum import Enum
from pydantic import BaseModel

class TaskStatus(str, Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    SUCCESS = 'success'
    FAILED = 'failed'

class TaskInfo(BaseModel):
    solution_name: str

class Task(BaseModel):
    status: TaskStatus
    progress: int
    info: TaskInfo

class TaskManager:
    def __init__(self):
        self.tasks: dict[str, Task] = {}  # key: task_id, value: Task
        self.lock = threading.Lock()

    def create_task(self, info=None):
        with self.lock:
            task_id = f"clone_{time.time()}"
            self.tasks[task_id] = Task(status=TaskStatus.PENDING, progress=0, info=info)
            return task_id

    def update_task(self, task_id, status=None, progress=None, info=None):
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            if status:
                task.status = status
            if progress is not None:
                task.progress = progress
            if info is not None:
                task.info = info
            return True

    def get_task(self, task_id):
        with self.lock:
            return self.tasks.get(task_id)
        
    def get_task_progress(self, task_id):
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return -1
            return task.progress

    def list_tasks(self):
        with self.lock:
            return dict(self.tasks)

task_manager = TaskManager()