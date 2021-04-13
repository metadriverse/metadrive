def _build_detector_thread():
    from direct.stdpy import threading
    from six.moves import queue

    class WorkerThread(threading.Thread):
        REQUEST = 1

        def __init__(self, cls, *args, **kwargs):
            super(WorkerThread, self).__init__()
            self.worker = cls(*args, **kwargs)
            self.inqueue = queue.Queue(maxsize=1)
            self.outqueue = queue.Queue()
            self.stopped = False

        def run(self):
            while not self.stopped:
                self.step()

        def step(self):
            func, args, kwargs = self.inqueue.get(timeout=300)
            assert hasattr(self.worker, func)
            result = getattr(self.worker, func)(*args, **kwargs)
            self.outqueue.put(result)

    return WorkerThread


class RemoteWorker:
    def __init__(self, cls, *args, **kwargs):
        DetectorThread = _build_detector_thread()
        self._async_worker = DetectorThread(cls, *args, **kwargs)
        self._async_worker.start()
        self.working = False

    def request(self, func_name, *args, **kwargs):
        self._async_worker.inqueue.put((func_name, args, kwargs))
        self.working = True
        # print("Successfully request function: {}".format(func_name))

    def get(self):
        assert self.working, "You should request first!"
        result = self._async_worker.outqueue.get(block=True)
        # print("Successfully get result of type: {}\n\n\n\n".format(type(result)))
        self.working = False
        return result

    def __del__(self):
        self._async_worker.stopped = True


if __name__ == '__main__':

    class TestWorker:
        def __init__(self, name):
            self.name = name

        def get_result(self, a, kw):
            return "Hi! {} {}-{}".format(self.name, a, kw)

    rd = RemoteWorker(TestWorker, "Bob")
    rd.request("get_result", 111, kw='CCC')
    print(rd.get())
