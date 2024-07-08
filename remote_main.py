import os
import sys
import datetime

if __name__ == '__main__':
    cur_datetime = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M')

    filename_res = os.path.join("results", cur_datetime, "console_out.txt")
    filename_pid = os.path.join("results", cur_datetime, "pid.txt")

    os.makedirs(os.path.dirname(filename_res), exist_ok=True)

    os.system(
        f"nohup {sys.executable} main.py --folder_name {cur_datetime} > {filename_res} 2>&1 & echo $! > {filename_pid}")
