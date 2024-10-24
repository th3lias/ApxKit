import os
import sys
import datetime

if __name__ == '__main__':
    cur_datetime = datetime.datetime.now().strftime('%d_%m_%Y_%H_%M')

    filename = "console_out.txt"

    filename = os.path.join("results", cur_datetime, filename)

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    os.system(f"nohup {sys.executable} main.py --folder_name {cur_datetime} > {filename} 2>&1 &")
