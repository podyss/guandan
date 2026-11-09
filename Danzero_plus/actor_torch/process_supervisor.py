import os
import signal
import subprocess
import time


def signal_process_group(process, signum):
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def stop_process(process, process_group=False, timeout=10):
    if isinstance(process, subprocess.Popen):
        if process.poll() is not None and not process_group:
            return
        if process_group:
            signal_process_group(process, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if process_group:
                signal_process_group(process, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
        if process_group:
            signal_process_group(process, signal.SIGKILL)
        return

    if not process.is_alive():
        process.join(timeout=0)
        return
    process.terminate()
    process.join(timeout=timeout)
    if process.is_alive():
        process.kill()
        process.join()


def stop_processes(processes, process_group=False, timeout=10):
    processes = list(reversed(list(processes)))
    if process_group and all(isinstance(process, subprocess.Popen) for process in processes):
        for process in processes:
            signal_process_group(process, signal.SIGTERM)
        deadline = time.monotonic() + timeout
        for process in processes:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                signal_process_group(process, signal.SIGKILL)
                process.wait()
            signal_process_group(process, signal.SIGKILL)
        return
    for process in processes:
        stop_process(process, process_group=process_group, timeout=timeout)
