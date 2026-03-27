import os
import logging
from abc import ABC
from omegaconf import DictConfig

import time
import hydra
import torch
from torch.utils.tensorboard import SummaryWriter

from fpin.fpin import eval_model, train_model
from fpin.VRPModel_attn_new import VRP_Net
from fpin.VRP_Loss1 import VRPLoss
from data.cvrp_dataset import CVRPDataset
from runners import BaseConstructionRunner

logger = logging.getLogger(__name__)

DATA_CLASS = {
    'CVRP': CVRPDataset
}


class Runner(BaseConstructionRunner, ABC):
    """wraps setup, training, testing of respective model
        experiments according to cfg"""

    def __init__(self, cfg: DictConfig):

        super(Runner, self).__init__(cfg)

        # fix path aliases changed by hydra
        # self.cfg = update_path(cfg)
        # OmegaConf.set_struct(self.cfg, False)

        # Model acronym
        # option for construction models to run with local search on top
        self.acronym, self.acronym_ls = self.get_acronym(model_name="fpin")

        # Name to identify run
        self.run_name = "{}_{}".format(self.cfg.run_type, self.acronym, time.strftime("%Y%m%dT%H%M%S"))

    def run(self):
        """Run experiment according to specified run_type."""
        if self.cfg.run_type in ['train', 'debug', 'resume']:
            self.setup(compatible_problems=DATA_CLASS)
        if self.cfg.run_type in ['train', 'debug']:
            self.train()
        elif self.cfg.run_type == 'resume':
            self.resume()
        elif self.cfg.run_type in ['val', 'test']:
            self.test()
        else:
            raise ValueError(f"unknown run_type: '{self.cfg.run_type}'. "
                             f"Must be one of ['train', 'resume', 'val', 'test', 'debug']")

    def _build_model(self):
        """Infer and set the model/model arguments provided to the learning algorithm."""
        cfg = self.cfg.copy()
        print('cfg.model_cfg.model_args', cfg.model_cfg.model_args)
        # load fpin model object
        self.model = VRP_Net(**cfg.model_cfg.model_args).to(self.device)
        if cfg.run_type in ["train", "debug", "resume"]:
            # print('os.getcwd()', os.getcwd())
            if cfg.cuda and torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
                # self.model = DDP(self.model)
            elif cfg.run_type == "resume":
                # need to update state_dct from resuming chkpt and update nr of epochs
                pass
            logger.info(f'model_args', cfg.model_cfg.model_args)
            logger.info(f'loss_cfg', cfg.loss_cfg)
            self.loss = VRPLoss(**cfg.loss_cfg).to(self.device)
            # .starts_weight, cfg.loss_cfg.pen_w, cfg.loss_cfg.load_w, cfg.loss_cfg.simple_loss,
            # cfg.loss_cfg.no_perms, cfg.loss_cfg.size_average, cfg.loss_cfg.with_penalty, cfg.loss_cfg.with_loads_loss
        else:
            # loads model with arguments from state_dct in checkpoint and sets model to eval
            logger.info(f"Loading model for {self.acronym} on {self.device}...")
            # self.model, _ = load_model_search(self.cfg.test_cfg.checkpoint_load_path)
            if self.device == torch.device("cpu"):
                try:
                    self.model.load_state_dict(torch.load(cfg.test_cfg.checkpoint_load_path,
                                                          map_location=torch.device('cpu'))['model'])
                except KeyError:
                    self.model.load_state_dict(torch.load(cfg.test_cfg.checkpoint_load_path,
                                                          map_location=torch.device('cpu')))
            else:
                try:
                    self.model.load_state_dict(torch.load(cfg.test_cfg.checkpoint_load_path)['model'])
                except KeyError:
                    self.model.load_state_dict(torch.load(cfg.test_cfg.checkpoint_load_path))
                # load fpin model object
                # self.model = VRP_Net(**cfg.model_cfg.model_args)
                # self.model.load_state_dict(torch.load(cfg.test_cfg.checkpoint_load_path))
            self.model.to(self.device)
            self.model.eval()
        if self.device in ["cuda", "mps"] and self.debug:
            logger.info(f"Used up GPU Mem. for loading model:")
            print_gpu_utilization()

    def _run_model(self):
        self.cfg.eval_opts_cfg["per_instance_time_limit_ls"] = int(self.per_instance_time_limit_constr)
        # print("self.cfg.eval_opts_cfg['per_instance_time_limit_ls']", self.cfg.eval_opts_cfg['per_instance_time_limit_ls'])
        self.cfg.eval_opts_cfg["post_process"] = False # ensure no post process during model-run
        print('self.cfg.eval_opts_cfg', self.cfg.eval_opts_cfg)
        return eval_model(model=self.model,
                          data_rp=self.ds.data,
                          normalised_data=self.cfg.normalize_data,
                          problem_size=self.cfg.graph_size,
                          problem=self.cfg.problem,
                          batch_size=self.cfg.test_cfg.eval_batch_size,
                          device=self.device,
                          opts=self.cfg.eval_opts_cfg)

    def train(self, **kwargs):
        """Train the specified model."""

        cfg = self.cfg.copy()
        # Optionally configure tensorboard
        tb_logger = None
        # if cfg.tb_logging:
        #     tb_logger = TbLogger(
        #         os.path.join(cfg.tb_log_path, "{}_{}".format(cfg.problem, cfg.graph_size), self.run_name))

        optimizer = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad],
                                     lr=cfg.train_cfg.lr, weight_decay=cfg.train_cfg.wd)

        logger.info(f"start training on {self.device}...")
        # epochs_done, best_loss, best_model, validation_loss_eps, training_loss_eps, CapViolation_eps,
        # CapViolation_mean_eps, CapViolation_mean_eps_v
        # Initialize once (maybe in your main training setup)
        writer = SummaryWriter(log_dir=cfg.train_cfg.tensorboard_logdir)
        summary, trained_model, logs = train_model(
            model=self.model,
            problem=cfg.problem,
            loss=self.loss,
            optimizer=optimizer,
            graph_size=cfg.graph_size,
            rp_train_data=self.local_train_set,
            device=self.device,
            resume=False,
            max_epochs=cfg.train_cfg.n_epochs,
            opts=cfg.train_cfg,
            rp_data_class=self.ds,  # from which to sample each epoch
            # val_dataset=self.val_data,  # fixed
            ckpt_save_path=cfg.checkpoint_save_path,
            env_cfg=cfg.env_kwargs,
            writer=writer
        )

        logger.info(f"training finished.")
        logger.info(f"Saving best model...")

        torch.save(
            {
                'model': trained_model,
                'optimizer': optimizer.state_dict(),
                'epoch': summary['epochs']
            },
            os.path.join(cfg.train_cfg.save_dir, 'best-epoch-{}.pt'.format(summary['epochs'])),
        )

        if writer is not None:
            writer.close()

        logger.info(f"summary results: {summary}")
        # logger.info(results)
        # solutions, summary = eval_rp(solutions, problem=self.cfg.problem)
        # self.save_results({
        #    "solutions": solutions,
        #    "summary": summary
        # })
        # logger.info(summary)

    def resume(self):
        """Resume training from a saved checkpoint."""
        cfg = self.cfg.copy()

        ckpt_path = cfg.test_cfg.checkpoint_load_path
        if ckpt_path is None:
            raise ValueError("No checkpoint_load_path provided for resume run.")

        logger.info(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        print('ckpt["epoch"]', ckpt["epoch"])
        # load model weights
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt["model"])

        # optimizer must be recreated before loading state
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.train_cfg.lr,
            weight_decay=cfg.train_cfg.wd
        )

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

        start_epoch = int(ckpt.get("epoch", 0))
        logger.info(f"Resuming training from epoch {start_epoch}")

        writer = SummaryWriter(log_dir=cfg.train_cfg.tensorboard_logdir)

        summary, trained_model, logs = train_model(
            model=self.model,
            problem=cfg.problem,
            loss=self.loss,
            optimizer=optimizer,
            graph_size=cfg.graph_size,
            rp_train_data=self.local_train_set,
            device=self.device,
            resume=True,
            start_epoch=start_epoch,
            max_epochs=cfg.train_cfg.n_epochs,
            opts=cfg.train_cfg,
            rp_data_class=self.ds,
            ckpt_save_path=cfg.checkpoint_save_path,
            env_cfg=cfg.env_kwargs,
            writer=writer
        )

        logger.info("resume training finished.")
        logger.info("Saving resumed best model...")

        torch.save(
            {
                'model': trained_model,
                'optimizer': optimizer.state_dict(),
                'epoch': summary['epochs']
            },
            os.path.join(cfg.train_cfg.save_dir, f'best-epoch-{summary["epochs"]}.pt'),
        )

        if writer is not None:
            writer.close()

        logger.info(f"summary results: {summary}")

    def test(self):
        """Test (evaluate) the trained model on specified dataset."""
        assert self.cfg.problem.upper() in ["CVRP"], "Only CVRP implemented currently"
        self.setup(compatible_problems=DATA_CLASS)

        results, summary = self.run_test()

    def get_acronym(self, model_name):
        acronym, acronym_ls = model_name, None
        if self.cfg.run_type in ["val", "test"]:
            if self.cfg.test_cfg.add_ls:
                ls_policy = str(self.cfg.test_cfg.ls_policy_cfg.local_search_strategy).upper()
                acronym_ls = ''.join([word[0] for word in ls_policy.split("_")])
                # acronym_ls = 'GORT_' + str(self.cfg.test_cfg.ls_policy_cfg.local_search_strategy).upper()
                if not self.cfg.eval_opts_cfg.post_process:
                    acronym = model_name + '_greedy' + acronym_ls
                else:
                    acronym = model_name + '_post' + acronym_ls
            elif self.cfg.test_cfg.use_distill:
                acronym_ls = 'HGS_decode'
                if not self.cfg.eval_opts_cfg.post_process:
                    acronym = model_name + '_' + acronym_ls
                else:
                    acronym = model_name + '_post' + acronym_ls
            else:
                if not self.cfg.eval_opts_cfg.post_process:
                    acronym = model_name + '_greedy'
                else:
                    acronym = model_name + '_post'
        return acronym, acronym_ls

    # @staticmethod
    # def seed_all(seed: int):
    #     """Set seed for all pseudo random generators."""
    #     # will set some redundant seeds, but better safe than sorry
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)

    def _update_path(self, cfg: DictConfig):
        """Correct the path to data files and checkpoints, since CWD is changed by hydra."""
        cwd = hydra.utils.get_original_cwd()

        if 'data_file_path' in list(cfg.test_cfg.keys()) and cfg.test_cfg.data_file_path is not None:
            cfg.test_cfg.data_file_path = os.path.normpath(
                os.path.join(cwd, cfg.test_cfg.data_file_path)
            )
        # if cfg.val_env_cfg.data_file_path is not None:
        #     cfg.val_env_cfg.data_file_path = os.path.normpath(
        #         os.path.join(cwd, cfg.val_env_cfg.data_file_path)
        #     )
        # if cfg.tester_cfg.test_env_cfg.data_file_path is not None:
        #     cfg.tester_cfg.test_env_cfg.data_file_path = os.path.normpath(
        #         os.path.join(cwd, cfg.tester_cfg.test_env_cfg.data_file_path)
        #     )

        if cfg.test_cfg.saved_res_dir is not None:
            cfg.test_cfg.saved_res_dir = os.path.normpath(
                os.path.join(cwd, cfg.test_cfg.saved_res_dir)
            )

        if cfg.test_cfg.checkpoint_load_path is not None:
            cfg.test_cfg.checkpoint_load_path = os.path.normpath(
                os.path.join(cwd, cfg.test_cfg.checkpoint_load_path)
            )
        # if cfg. in ["train", "resume"]:

        if cfg.run_type in ["train", "resume"]:
            # if cfg.train_cfg.model_load.path is not None:
            #     cfg.train_cfg.model_load.path = os.path.normpath(
            #         os.path.join(cwd, cfg.train_cfg.model_load.path)
            #     )
            # if cfg.train_cfg.local_target_path is not None:
            #     cfg.train_cfg.local_target_path = os.path.normpath(
            #         os.path.join(cwd, cfg.train_cfg.local_target_path)
            #     )
            print('cfg.train_cfg.train_dataset', cfg.train_cfg.train_dataset)
            if cfg.train_cfg.train_dataset is not None:
                cfg.train_cfg.train_dataset = os.path.normpath(
                    os.path.join(cwd, cfg.train_cfg.train_dataset)
                )
            print('cfg.train_cfg.train_dataset', cfg.train_cfg.train_dataset)
            if cfg.env_kwargs.generator_args.single_large_instance is not None:
                cfg.env_kwargs.generator_args.single_large_instance = os.path.normpath(
                    os.path.join(cwd, cfg.env_kwargs.generator_args.single_large_instance)
                )
        return cfg

# def update_path(cfg: DictConfig):
#     """Correct the path to data files and checkpoints, since CWD is changed by hydra."""
#     cwd = hydra.utils.get_original_cwd()
#
#     if 'data_file_path' in list(cfg.keys()):
#         cfg.data_file_path = os.path.normpath(
#             os.path.join(cwd, cfg.data_file_path)
#         )
#     # if cfg.val_env_cfg.data_file_path is not None:
#     #     cfg.val_env_cfg.data_file_path = os.path.normpath(
#     #         os.path.join(cwd, cfg.val_env_cfg.data_file_path)
#     #     )
#     # if cfg.tester_cfg.test_env_cfg.data_file_path is not None:
#     #     cfg.tester_cfg.test_env_cfg.data_file_path = os.path.normpath(
#     #         os.path.join(cwd, cfg.tester_cfg.test_env_cfg.data_file_path)
#     #     )
#
#     if cfg.test_cfg.saved_res_dir is not None:
#         cfg.test_cfg.saved_res_dir = os.path.normpath(
#             os.path.join(cwd, cfg.test_cfg.saved_res_dir)
#         )
#
#     if cfg.test_cfg.checkpoint_load_path is not None:
#         cfg.test_cfg.checkpoint_load_path = os.path.normpath(
#             os.path.join(cwd, cfg.test_cfg.checkpoint_load_path)
#         )
#     return cfg
#
#
# def remove_dir_tree(root: str, pth: Optional[str] = None):
#     """Remove the full directory tree of the root directory if it exists."""
#     if not os.path.isdir(root) and pth is not None:
#         # select root directory from path by dir name
#         i = pth.index(root)
#         root = pth[:i + len(root)]
#     if os.path.isdir(root):
#         shutil.rmtree(root)


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
