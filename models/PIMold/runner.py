"""Runner for the original Thyssens-2022 "old PIM" baseline.

Subclasses the F-PIN ``Runner`` and overrides ONLY model construction and the model-run
call. Path handling, dataset loading from ``data_file_path``, RPSolution scoring,
cost_v / fleet-violation metrics and logging are all inherited unchanged, so the old PIM
baseline is evaluated on an identical footing to F-PIN.
"""
import logging

import torch

from runner import Runner
from models.PIMold.pimold import eval_model_pimold, build_old_model

logger = logging.getLogger(__name__)


class PIMoldRunner(Runner):
    """Evaluate the original (softassign) permutation-invariant PIM model as a baseline."""

    def __init__(self, cfg):
        super(PIMoldRunner, self).__init__(cfg)
        # override the inherited "fpin" acronym for output naming / logging
        self.acronym = "PIMold"

    def run(self):
        if self.cfg.run_type not in ["val", "test"]:
            raise ValueError(
                f"PIMold baseline only supports run_type in ['val','test'] "
                f"(got '{self.cfg.run_type}'). Training the old model is out of scope; "
                f"we only evaluate released checkpoints."
            )
        self.test()

    def _build_model(self):
        """Instantiate the OLD softassign PIM model and load its released checkpoint."""
        cfg = self.cfg.copy()

        # size-aware construction (VRP20/50/100 differ in layers/main_dim; VRP100 uses the
        # memory-modified variant) — see models/PIMold/pimold.OLD_MODEL_SPECS.
        self.model = build_old_model(int(cfg.graph_size), self.device)

        # old .pth is a bare state_dict or a {'model','optimizer','loss'} dict
        map_loc = torch.device("cpu") if self.device == torch.device("cpu") else None
        ckpt = torch.load(cfg.test_cfg.checkpoint_load_path, map_location=map_loc)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            logger.warning(
                f"[PIMold] load_state_dict non-strict: "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys"
            )
        self.model.to(self.device)
        self.model.eval()

    def _run_model(self):
        self.cfg.eval_opts_cfg["post_process"] = False
        return eval_model_pimold(
            model=self.model,
            data_rp=self.ds.data,
            normalised_data=self.cfg.normalize_data,
            problem_size=self.cfg.graph_size,
            problem=self.cfg.problem,
            device=self.device,
            opts=self.cfg.eval_opts_cfg,
        )
