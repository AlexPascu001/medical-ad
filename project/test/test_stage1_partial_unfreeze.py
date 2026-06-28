import unittest

import torch.nn as nn

from model import DINOv3Backbone
from train import Trainer


class _FakeEncoder(nn.Module):
    def __init__(self, block_count: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(block_count)])
        self.norm = nn.LayerNorm(4)


class _FakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = DINOv3Backbone.__new__(DINOv3Backbone)
        nn.Module.__init__(backbone)
        backbone.backbone = _FakeEncoder()
        backbone.num_blocks = len(backbone.backbone.blocks)
        backbone.freeze_backbone = False
        self.backbone = backbone


def _make_schedule_trainer() -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.model = _FakeDetector()
    trainer.stage1_schedule = {
        'enabled': True,
        'head_warmup_epochs': 5,
        'unfreeze_last_n_blocks': 2,
    }
    trainer._stage1_backbone_trainable = None
    return trainer


class Stage1PartialUnfreezeTests(unittest.TestCase):
    def test_stage1_schedule_freezes_all_encoder_parameters_during_head_warmup(self):
        trainer = _make_schedule_trainer()
        trainer.epoch = 0

        trainer._apply_stage1_backbone_schedule()

        wrapper = trainer.model.backbone
        self.assertTrue(wrapper.freeze_backbone)
        self.assertTrue(all(not parameter.requires_grad for parameter in wrapper.backbone.parameters()))

    def test_stage1_schedule_only_unfreezes_final_blocks_and_norm_after_warmup(self):
        trainer = _make_schedule_trainer()
        trainer.epoch = 5

        trainer._apply_stage1_backbone_schedule()

        wrapper = trainer.model.backbone
        selected = wrapper.get_partial_backbone_parameters(last_n_blocks=2)
        selected_ids = {id(parameter) for parameter in selected}

        self.assertFalse(wrapper.freeze_backbone)
        self.assertTrue(all(
            parameter.requires_grad == (id(parameter) in selected_ids)
            for parameter in wrapper.backbone.parameters()
        ))
        self.assertFalse(wrapper.backbone.blocks[0].training)
        self.assertTrue(wrapper.backbone.blocks[-1].training)


if __name__ == '__main__':
    unittest.main()
