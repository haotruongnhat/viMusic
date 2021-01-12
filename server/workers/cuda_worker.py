import redis
from py3nvml.py3nvml import *
import mpipe
import json

class CudaWorker(mpipe.UnorderedWorker):
    """
    A Base class to initialize connection, maintaining lock, and choosing memory
    """
    def __init__(self,global_lock=None,pool=None,allow_memory=None):
        self._global_lock = global_lock

        if pool is not None:
            self._db = redis.StrictRedis(connection_pool=pool)
        else:
            self._db = None

        self._allow_memory = allow_memory

        self._cuda_device_index = None

    def init_cuda_memory(self):
        if self._allow_memory is not None:
            nvmlInit()
            #Check available smallest unused gpu devices
            mem = float('inf')
            for index in range(nvmlDeviceGetCount()):
                handle = nvmlDeviceGetHandleByIndex(index)
                info = nvmlDeviceGetMemoryInfo(handle)
                unused = (info.total - info.used) >> 20
                if unused > self._allow_memory:
                    if unused < mem:
                        mem = unused
                        self._cuda_device_index = index

        if self._cuda_device_index is None:
            raise ValueError("Not enough memory to allocate")

    def update_progress(self,query):
        if self._db is not None:
            self.acquire()
            print(json.dumps({
                'idProcess': query["idProcess"],
                "status" : "wait",
                "percentage" : query["percentage"]
            }))
            self._db.set(query["idProcess"], json.dumps({
                'idProcess': query["idProcess"],
                "status" : "wait",
                "percentage" : query["percentage"]
            }))
            self.release()
        else:
            print("Local mode: Progress will not be sent")
        
    def acquire(self):
        if self._global_lock is not None:
            self._global_lock.acquire()

    def release(self):
        if self._global_lock is not None:
            self._global_lock.release()