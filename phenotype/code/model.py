import os
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# Default summary path for best hyperparameters
_DEFAULT_SUMMARY_PATH = os.path.join('spore_optuna', 'sporo_full_std_v2_cont_exp_sporulation', 'summary.txt')


def load_best_hparams_from_summary(summary_path: str = _DEFAULT_SUMMARY_PATH) -> dict:
    """Load key=value pairs from Optuna summary.txt and parse types.

    Returns a dict with parsed scalars (int/float/bool/str).
    Missing file yields empty dict.
    """
    hparams: dict = {}
    try:
        if not os.path.exists(summary_path):
            return hparams
        with open(summary_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or '=' not in line:
                    continue
                key, raw_val = line.split('=', 1)
                key = key.strip()
                val_str = raw_val.strip()
                # Parse bool
                low = val_str.lower()
                if low in ('true', 'false'):
                    hparams[key] = (low == 'true')
                    continue
                # Parse int/float
                try:
                    # Support scientific notation
                    if any(ch in val_str for ch in ('.', 'e', 'E')):
                        hparams[key] = float(val_str)
                    else:
                        hparams[key] = int(val_str)
                    continue
                except Exception:
                    pass
                # Fallback to string
                hparams[key] = val_str
    except Exception:
        # Be resilient: return what we have
        return hparams
    return hparams

class SporulationModel(nn.Module):
    """Parametric 1D CNN that automatically loads best HPs from summary.txt.

    The architecture mirrors the Optuna-tuned `SporoCNN` used during HPO.
    It builds 1-3 small conv blocks according to loaded hyperparameters and
    uses the same rounding rules as in tuning to derive integer channels and
    kernel sizes from the sampled continuous values.
    """

    def __init__(self, summary_path: str = _DEFAULT_SUMMARY_PATH, num_classes: int = 2, params: Optional[Dict[str, Any]] = None):
        super(SporulationModel, self).__init__()

        # Load HPs and derive architecture. Prefer explicit Optuna params if provided.
        hp = {}
        if isinstance(params, dict) and len(params) > 0:
            hp = dict(params)
        else:
            hp = load_best_hparams_from_summary(summary_path)

        # Derive integers using the same logic as tuning
        k1_idx = int(hp.get('k1_idx', 100))
        k1 = 2 * k1_idx + 1
        c1_cont = float(hp.get('c1_cont', 64.0))
        c1 = int(max(16, min(128, 16 * round(c1_cont / 16.0))))
        stride1 = max(1, int(round(float(hp.get('stride1_cont', 10.0)))))
        pool1_k = max(2, int(round(float(hp.get('pool1_k_cont', 50.0)))))
        pool1_s = max(1, int(round(float(hp.get('pool1_s_cont', 25.0)))))

        n_blocks = int(hp.get('n_blocks', 2))
        n_blocks = max(1, min(3, n_blocks))
        k_small_idx = int(hp.get('k_small_idx', 2))
        k_small = 2 * k_small_idx + 1
        c2_cont = float(hp.get('c2_cont', 128.0))
        c2 = int(max(32, min(256, 32 * round(c2_cont / 32.0))))
        c3_cont = float(hp.get('c3_cont', 256.0))
        c3 = int(max(64, min(512, 32 * round(c3_cont / 32.0))))
        use_pool2 = bool(hp.get('use_pool2', True))
        use_pool3 = bool(hp.get('use_pool3', False)) if n_blocks >= 2 else False
        # Pooling kernel/stride indices follow the exact rules used in SporoCNN during tuning
        if use_pool2:
            pool2_k = int(2 * int(hp.get('pool2_k_idx', 3)) + 1)
            pool2_s = int(hp.get('pool2_s_int', 2))
        else:
            pool2_k = 3
            pool2_s = 2
        if n_blocks >= 3 and use_pool3:
            pool3_k = int(2 * int(hp.get('pool3_k_idx', 2)) + 1)
            pool3_s = int(hp.get('pool3_s_int', 2))
        else:
            pool3_k = 3
            pool3_s = 2

        drop1 = float(hp.get('drop1', 0.1))
        drop2 = float(hp.get('drop2', 0.1))
        drop3 = float(hp.get('drop3', 0.1))
        drop_fc = float(hp.get('drop_fc', 0.3))
        fc_hidden_cont = float(hp.get('fc_hidden_cont', 256.0))
        fc_hidden = int(max(64, min(1024, 32 * round(fc_hidden_cont / 32.0))))
        self.act1_name = str(hp.get('act1', 'relu'))
        global_pool = str(hp.get('global_pool', 'avg'))

        # Conv1
        p1 = (k1 - 1) // 2
        self.conv1 = nn.Conv1d(4, c1, kernel_size=k1, stride=stride1, padding=p1, bias=True)
        self.bn1 = nn.BatchNorm1d(c1)
        self.drop1 = nn.Dropout(drop1)
        self.pool1 = nn.MaxPool1d(kernel_size=pool1_k, stride=pool1_s)

        # Small conv blocks
        channels_in = c1
        blocks = []
        out_dims = [c2, c3, max(c3, 256)]
        pool_cfg = [
            (use_pool2, pool2_k, pool2_s),
            (use_pool3, pool3_k, pool3_s),
            (False, 1, 1),
        ]
        drops = [drop2, drop3, drop3]

        for i in range(n_blocks):
            co = out_dims[i]
            p_small = (k_small - 1) // 2
            blocks.append(nn.Conv1d(channels_in, co, kernel_size=k_small, padding=p_small, bias=True))
            blocks.append(nn.BatchNorm1d(co))
            blocks.append(nn.ReLU(inplace=True))
            blocks.append(nn.Dropout(drops[i]))
            use_pool, pk, ps = pool_cfg[i]
            if use_pool:
                blocks.append(nn.MaxPool1d(kernel_size=pk, stride=ps))
            channels_in = co

        self.blocks = nn.Sequential(*blocks)
        self.gpool = nn.AdaptiveAvgPool1d(1) if global_pool == 'avg' else nn.AdaptiveMaxPool1d(1)

        self.fc_drop = nn.Dropout(drop_fc)
        self.fc1 = nn.Linear(channels_in, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def _act1(self, x: torch.Tensor) -> torch.Tensor:
        if self.act1_name == 'exp':
            return torch.exp(x)
        if self.act1_name == 'gelu':
            return F.gelu(x)
        if self.act1_name == 'softplus':
            return F.softplus(x)
        return F.relu(x)

    def forward(self, x):
        # Conv1 block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self._act1(x)
        x = self.drop1(x)
        x = self.pool1(x)

        # Small conv blocks
        x = self.blocks(x)

        # Head
        x = self.gpool(x)
        x = x.squeeze(-1)
        x = self.fc_drop(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
