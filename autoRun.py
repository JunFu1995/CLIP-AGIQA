import pynvml 
import time 
import subprocess

pynvml.nvmlInit()


def taskAux():
    cmd_list = []
    for m in ['AGIQA_clean']:
        for ds in ['AGIQA3k']: #,'AGIQA2023'
            cmd = 'nohup python train_test_clip_auxiliary.py --dataset %s --model %s &' % (ds, m)
            cmd_list.append(cmd)
    return cmd_list

cmd = taskAux()

while cmd:
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    memoinfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    if memoinfo.used / 1e6 < 14 * 1000:
        c = cmd.pop(0)
        subprocess.call(c, shell=True)
        time.sleep(15)
    else:
        time.sleep(300)




