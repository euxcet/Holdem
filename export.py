import time
import os
import shutil

while True:
    root = "/home/zhouchengchi/ray_results"
    for run in sorted(os.listdir(root), reverse=True):
        if run.startswith('PPO'):
            run_folder = os.path.join(root, run)
            for x in os.listdir(run_folder):
                if x.startswith('PPO'):
                    run_folder = os.path.join(run_folder, x)
                    break
            for checkpoint in sorted(os.listdir(run_folder), reverse=True):
                if checkpoint.startswith('checkpoint'):
                    cid = int(checkpoint.split('_')[-1])
                    print('Current epoch:', cid)
                    if cid % 10 == 0:
                        print('Export', checkpoint)
                        run_folder = os.path.join(run_folder, checkpoint, 'policies', 'learned')
                        shutil.copyfile(os.path.join(run_folder, 'model.pt'), 'checkpoint/test/model.pt')
                        shutil.copyfile(os.path.join(run_folder, 'model.pt'), 'checkpoint/test/model_' + str(cid) + '.pt')
                    break
            break
    time.sleep(60)
