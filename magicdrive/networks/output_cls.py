from dataclasses import dataclass
from typing import Tuple

import torch
import torch_npu

from diffusers.utils import BaseOutput


@dataclass
class BEVControlNetOutput(BaseOutput):
    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor
    encoder_hidden_states_with_cam: torch.Tensor
