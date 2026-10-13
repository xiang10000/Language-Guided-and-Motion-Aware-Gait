"""Dataset-free smoke test: build LMGait from the SUSTech1K yaml and run forward
passes on random tensors. Run from the repo root:

    .venv\\Scripts\\python.exe check_forward.py

Needs ViT-B-16.pt, bpe_simple_vocab_16e6.txt.gz and
pretrained_model/dinov2_vits14_pretrain.pth, but no dataset.
"""
import json
import os
import sys
import tempfile

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'opengait'))
os.chdir(ROOT)

from utils import config_loader, get_msg_mgr  # noqa: E402
from modeling.models.LMGait import LMGait  # noqa: E402

N, S = 2, 8  # sequences, frames per sequence


def main():
    # file:// rendezvous avoids TCPStore, which fails on some Windows setups
    store_file = os.path.join(tempfile.mkdtemp(), 'store').replace(os.sep, '/')
    torch.distributed.init_process_group(
        'gloo', init_method='file:///' + store_file, rank=0, world_size=1)

    tmp = tempfile.mkdtemp()
    part = os.path.join(tmp, 'part.json')
    with open(part, 'w') as f:
        json.dump({'TRAIN_SET': [str(i) for i in range(10)], 'TEST_SET': []}, f)

    cfgs = config_loader('configs/LMGait/LMGait_SUSTECH.yaml')
    cfgs['model_cfg']['dataset_partition'] = part
    get_msg_mgr().init_logger(os.path.join(tmp, 'log'), False)

    # Build the model without BaseModel.__init__, which would also need the dataset.
    model = LMGait.__new__(LMGait)
    nn.Module.__init__(model)
    model.msg_mgr = get_msg_mgr()
    model.cfgs = cfgs
    model.iteration = 0
    model.device = 0
    model.build_network(cfgs['model_cfg'])
    model.init_parameters()
    model.to(torch.device('cuda', 0))

    h, w = 256, 128
    sils = torch.rand(N, S, 3, h, w, device='cuda')
    ratios = torch.full((N, S, 1), 0.5, device='cuda')
    labs = torch.arange(N, device='cuda')

    for training in (True, False):
        model.train(training)
        model.backbone.eval()
        with torch.autocast('cuda', enabled=True):
            out = model(([sils, ratios], labs, None, None, None))
        emb = out['inference_feat']['embeddings']
        print(f"training={training}: embeddings {tuple(emb.shape)}")
        if training:
            for k, v in out['training_feat'].items():
                if not isinstance(v, dict):
                    print(f"  {k}: {float(v):.4f}")
            assert emb.shape[0] == N, 'batch dimension must be preserved'

    print('peak GPU memory: %.2f GB' % (torch.cuda.max_memory_allocated() / 1e9))
    print('OK')


if __name__ == '__main__':
    main()
