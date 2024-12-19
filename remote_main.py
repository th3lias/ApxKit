import os
import sys
import datetime

if __name__ == '__main__':
    cur_datetime = datetime.datetime.now()

    cur_datetime_string = cur_datetime.strftime('%d_%m_%Y_%H_%M_%S')

    filename_res = os.path.join("results", cur_datetime_string, "console_out.txt")
    filename_pid = os.path.join("results", cur_datetime_string, "pid.txt")

    os.makedirs(os.path.dirname(filename_res), exist_ok=True)

    os.system(
        f"nohup {sys.executable} main.py --folder_name {cur_datetime_string} > {filename_res} 2>&1 & echo $! > {filename_pid}")
