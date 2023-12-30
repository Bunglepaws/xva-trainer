import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.skip_nnmodule_hook_guards = False
torch.set_warn_always(False)

import argparse
import traceback
import gc
import torch
from server import between_callback
from server import models_manager


class DummySocket:
    @staticmethod
    async def send(msg: any):
        print(msg)


parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('--headless', action='store_true', required=False)
parser.add_argument('--dataset_path', required=False)
parser.add_argument('--output_path', required=False)
parser.add_argument('--use_amp', action='store_true', required=False)
parser.add_argument('--batch_size', default=8, type=int, required=False)
parser.add_argument('--num_workers', default=3, type=int, required=False)
parser.add_argument('--backup_every_x', default=2, type=int, required=False)
parser.add_argument('--lang', default='en', required=False)
parser.add_argument('--force_stage', default=None, required=False, choices=['1', '2', '3'])
args, _ = parser.parse_known_args()

message = {'model': '',
           'task': 'startTraining',
           'gpus': '0',
           'data': {
               'dataset_path': args.dataset_path,
               'output_path': args.output_path,
               'checkpoint': '[base]',
               'use_amp': 'true' if args.use_amp else 'false',
               'num_workers': args.num_workers,
               'batch_size': args.batch_size,
               'bkp_every_x': args.backup_every_x,
               'lang': args.lang,
               'rowElem': {},
           }}
if args.force_stage:
    message['data']['force_stage'] = args.force_stage
print(message)
model = message["model"]
# gpus = [int(g) for g in message["gpus"].split(",")] if "gpus" in message else [0]
gpus = [0]
task = message["task"] if "task" in message else None
data = message["data"] if "data" in message else None
# _thread.start_new_thread(between_callback, (models_manager, data, None, gpus, task == "resume"))
try:
    between_callback(models_manager, data, DummySocket(), gpus, task == 'resume')
except:
    print(traceback.format_exc())
print('gc.collect')
gc.collect()
torch.cuda.empty_cache()
