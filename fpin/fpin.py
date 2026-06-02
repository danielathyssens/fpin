import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import time
import os
import tracemalloc
import itertools
import random
import warnings
import logging
from tqdm import tqdm
from typing import Dict, Union, List, NamedTuple, Tuple, Any, Optional
from omegaconf import DictConfig
from formats import CVRPInstance, RPSolution
from data.cvrp_dataset import CVRPDataset
from fpin.data_utils.preprocess1 import prep_test_data
from fpin.utils_all.basic_funcs import move_to, zeros, to_variable
from fpin.VRPModel_attn_new import VRP_Net
from fpin.utils import DatasetVrp, DatasetVrp100, DatasetVrpFromNPZ, get_preliminaries, get_dataset
from fpin.utils import prep_data_fpin, make_valid_and_adjust, make_RPSolution, get_travel_costs
from fpin.utils_all.print_out import print_func
from fpin.utils_all.get_path import load_estimate_capacity
from fpin.utils_all.decoder_fallback import decode_vehicle_assignment_1, decode_vehicle_assignment_v2, decode_giant_tour_split_dp, decode_vehicle_assignment_faithful, decode_transition_following
from fpin.VRP_Loss1 import VRPLoss

from torch.utils.tensorboard import SummaryWriter

EPS = np.finfo(np.float32).eps
# limit for training if there's no val_loss improvement
NO_IMPRVMT_LIMIT = 100

logger = logging.getLogger(__name__)


def train_model(problem: str,
                graph_size: int,
                model: VRP_Net,
                loss: VRPLoss,
                optimizer: optim.Optimizer,
                opts: Union[DictConfig, NamedTuple],
                env_cfg: DictConfig,
                ckpt_save_path: str,
                rp_train_data: List[RPSolution] = None,
                rp_data_class: CVRPDataset = None,
                val_dataset: CVRPDataset = None,
                start_epoch: int = 0,
                resume: bool = False,
                max_epochs: int = 100,
                device: torch.device = torch.device("cuda"),
                writer: SummaryWriter = None
                ):
    # preliminaries (size-specific hyperparams)
    tracemalloc.start()
    nr_train_files, pen_w, load_w, starts_weight = get_preliminaries(graph_size, is_train=True)
    print('nr_train_files', nr_train_files)
    opts["nr_train_files"] = nr_train_files
    is_size_50 = False if graph_size != 50 else True
    stationary_epoch = 40
    distill_union = None
    warmup_epochs = getattr(opts, "warmup_epochs", 3)
    min_es_epoch = getattr(opts, "min_es_epoch", warmup_epochs + 8)

    # Route to the lazy npz Dataset whenever the train dir contains .npz files
    # (covers both legacy targets and the new HQ multi-key format that the
    # in-memory `load_in_get_dataset` path cannot consume).
    _td = opts.train_dataset
    _use_npz_dataset = ("npz" in _td) or (
        isinstance(_td, str)
        and os.path.isdir(_td)
        and any(fn.endswith(".npz") for fn in os.listdir(_td))
    )
    dataset = get_dataset(opts, rp_data_class, env_cfg,
                          fixed_train_targets=rp_train_data,
                          train_dat_load="load_in_Dataset" if _use_npz_dataset else "load_in_get_dataset",
                          nr_datapoints=opts.nr_train_samples)

    if isinstance(dataset[0], DatasetVrp) or isinstance(dataset[0], DatasetVrp100):
        training_generator = DataLoader(dataset[0], opts.batch_size, shuffle=True, num_workers=opts.num_workers)
        val_generator = DataLoader(dataset[1], opts.batch_size, shuffle=False, num_workers=opts.num_workers)
    elif isinstance(dataset[0], DatasetVrpFromNPZ):
        training_generator = DataLoader(dataset[0], opts.batch_size, shuffle=True, num_workers=opts.num_workers)
        val_generator = DataLoader(dataset[1], opts.batch_size, shuffle=False, num_workers=opts.num_workers)
    else:
        warnings.warn(f"Aborting... Unknown training dataset type: {type(dataset[0])}")
        training_generator, val_generator = None, None
        raise ModuleNotFoundError

    # Loop over epochs
    epochs_done, no_imprvmt_count, best_ema_loss, best_model = start_epoch, 0, 99999.0, None  # early stopping
    CapViolation_eps, CapViolation_mean_eps, CapViolation_mean_eps_v = [], [], []
    training_loss_eps, validation_loss_eps = [], []
    steps_per_epoch = len(training_generator)
    print('steps_per_epoch', steps_per_epoch)
    if epochs_done != 0:
        print(f"resuming training from {epochs_done} ...")
    min_es_epoch = max(min_es_epoch, stationary_epoch)
    for epoch in range(epochs_done, max_epochs):
        epochs_done += 1
        logger.info(f"EPOCH: {epoch}")

        data_prep_time_st = time.time()
        model.train()
        # train-step
        tr_loss_b, capa_v_b, capa_v_m_b = train_one_epoch(
            training_generator, opts, graph_size,
            model, loss, optimizer, device,
            writer=writer, epoch=epoch,
            steps_per_epoch=steps_per_epoch,
            time_st=data_prep_time_st, distill_union=distill_union
        )

        # val-step
        val_loss_b, val_capa_v_m_b, last_validation_values = validate_epoch(
            val_generator,
            model, loss, device,
            graph_size, opts,
            writer=writer,
            epoch=epoch,
            steps_per_epoch=steps_per_epoch,
            distill_union=distill_union, )

        # Check Memory Usage
        # if opts.g:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")

        # APPENDING EPOCH LOSS
        curr_val_loss = np.mean(val_loss_b)
        # optional smoothing (recommended)
        # (less lag): higher alpha tracks curr_val_loss more closely (0.4)
        # alpha = 0.70 → almost raw(barely smoothing)
        es_alpha = getattr(opts, "es_ema_alpha", 0.70)  # 0.05–0.2 typical
        if epoch == 0 or "val_ema" not in locals():
            val_ema = curr_val_loss
        else:
            val_ema = (1.0 - es_alpha) * val_ema + es_alpha * curr_val_loss
        logger.info(f"Current validation loss: {curr_val_loss}")
        validation_loss_eps.append(np.mean(val_loss_b))
        logger.info(f"Current train loss: {np.mean(tr_loss_b)}")
        training_loss_eps.append(np.mean(tr_loss_b))

        # if not all(v is None for v in capa_v_b):
        # APPENDING EPOCH VIOLATION
        # print('capa_v_b', capa_v_b)
        # print('capa_v_m_b', capa_v_m_b)
        CapViolation_eps.append(np.mean(capa_v_b))
        CapViolation_mean_eps.append(np.mean(capa_v_m_b))
        # if not all(v is None for v in val_capa_v_m_b):
        # APPENDING EPOCH VIOLATION FOR VALIDATION
        val_capa_vals = [v for v in val_capa_v_m_b if v is not None]
        CapViolation_mean_eps_v.append(float(np.mean(val_capa_vals)) if len(val_capa_vals) > 0 else float("nan"))
        #print('val_capa_v_m_b', val_capa_v_m_b)
        if epoch % 2 == 0:
            print('\nEpoch: {}, TRAIN loss (avg of all batches): {}'.format(epoch, np.mean(tr_loss_b)))
            print('\nEpoch: {}, VALID loss (avg of all batches): {}'.format(epoch, np.mean(val_loss_b)))
            if not opts.vrp_size == 200 and last_validation_values is not None:
                print_func(*last_validation_values, opts.load, opts.fleet_size)

        #  if not all(v is None for v in capa_v_b):
        #      print('Epoch: {},MEAN CAPA VIOL (avg of all batches): {}'.format(epoch,
        #                                                                       np.mean(violations_batch_mean)))
        #  if not all(v is None for v in capa_v_b):
        #      print('Epoch: {}, MEAN CAPA VIOL VAL (avg of all batches): {}'.format(epoch,
        #                                                                         np.mean(violations_batch_mean_v)))

        ################### END OF ONE EPOCH ###################
        # use EMA for early stop decision
        metric = float(f"{val_ema:.3f}")
        # metric = curr_val_loss  # <-- RAW val loss (no lagging)
        min_delta = getattr(opts, "es_min_delta", 1e-4)
        patience = getattr(opts, "es_patience", NO_IMPRVMT_LIMIT)
        if epoch == min_es_epoch:
            logger.info(f"Early stopping activated at epoch {epoch}")
        # simple early stopping
        if epoch >= min_es_epoch:
            improved = (best_ema_loss - metric) > min_delta
            if improved:
                best_ema_loss = metric
                best_model = model.state_dict()
                no_imprvmt_count = 0
                torch.save(
                    {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
                    os.path.join(opts.save_dir, 'best.pt')
                )
            else:
                no_imprvmt_count += 1
                print('no_imprvmt_count', no_imprvmt_count, 'best_ema_loss', best_ema_loss, 'metric', metric)

        # save ckpt:
        if epoch % opts.checkpoint_epochs == 0:
            print('Saving model and state...')
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                },
                os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epochs_done))
            )
        # if epoch >= min_es_epoch and curr_val_loss < best_ema_loss:
        #     torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch},
        #                os.path.join(opts.save_dir, 'best.pt'))

        # stop
        if epoch >= min_es_epoch and no_imprvmt_count >= patience:
            print('Early stopping. last best_ema_loss:', best_ema_loss)
            print('last metric:', metric, 'raw curr_val_loss:', curr_val_loss)
            break
        tracemalloc.stop()
    train_summary = {
        'epochs': epochs_done,
        'best_ema_val_loss': best_ema_loss,
        'last_val_loss': validation_loss_eps[-1],
        'last_train_loss': training_loss_eps[-1],
        'Capacity Violation': CapViolation_eps[-1],
        'Mean Capacity Violation (val)': CapViolation_mean_eps_v[-1]

    }
    return train_summary, best_model, (
        validation_loss_eps, training_loss_eps, CapViolation_eps, CapViolation_mean_eps, CapViolation_mean_eps_v)


def train_one_epoch(generator, opts, graph_size,
                    model, loss_f, optimizer, device, writer, epoch,
                    distill_union=None,
                    steps_per_epoch=None,
                    time_st=None):
    lambda_distill = getattr(opts, "lambda_distill", 0.03)
    device_for_loss = 'cuda:1' if torch.cuda.device_count() > 1 else device

    tr_loss_batches, capa_v_sums, capa_v_means = [], [], []
    time_st_ep = time.time()
    batch_idx = 0
    distill_loss = None
    regret_loss = None
    base_loss = None
    decode_rl_loss = None  # torch scalar or None
    decode_rl_loss_w = None  # torch scalar or None
    rl_debug = "not_run"  # string, always set

    # ---------------- TIMINGS (added) ----------------
    alpha = 0.05  # EMA smoothing
    t_prev_end = time.time()
    t_ema = {"data": 0.0, "move": 0.0, "fwd": 0.0, "loss": 0.0, "bwd": 0.0, "tot": 0.0}

    # ---------------- LOGGING ------------------
    # ---- running stats for epoch-level TB ----
    running = {
        "loss_sum": 0.0,
        "count": 0.0,
        "rl_loss_sum": 0.0,
        "rl_loss_w_sum": 0.0,
        "rl_obj_sum": 0.0,
        "rl_count": 0.0,
    }


    def _ema_update(key, val):
        t_ema[key] = (1 - alpha) * t_ema[key] + alpha * float(val)

    # -------------------------------------------------

    for batch, targets, target_loads, tr_id in generator:
        t_batch_start = time.time()
        edge_base, edge_rank = None, None
        impr_mean, impr_frac = None, None
        extras, extras_t = {
            "edge_base": 0.0,
            "edge_rank": 0.0,
            "edge_match": 0.0,
            "succ_ce": 0.0,
            "cover": 0.0,
            "antidepot": 0.0,
        }, None
        _ema_update("data", t_batch_start - t_prev_end)  # time waiting for DataLoader

        t0 = time.time()
        optimizer.zero_grad()

        # ---------------- TIMINGS: move/targets (added) ----------------
        t_move0 = time.time()
        # --------------------------------------------------------------

        # Move batch
        if torch.cuda.device_count() > 1:
            fleet_b, depot_b, custom_b, dem_b, dists_b = move_to(batch, device='cuda:0')
        else:
            fleet_b, depot_b, custom_b, dem_b, dists_b = move_to(batch, device)
        # Targets
        targets = prepare_targets(targets, graph_size, batch, device)
        targets, target_loads = to_variable(targets, target_loads, device=device_for_loss)

        # ---------------- TIMINGS: move/targets end (added) ----------------
        _ema_update("move", time.time() - t_move0)
        t_fwd0 = time.time()
        # ------------------------------------------------------------------

        # Model
        vrp_logits, edge_probs = model(depot_b, custom_b, fleet_b, dem_b, dists_b, sample=False)
        # with torch.no_grad():
        #     row_sum = vrp_logits.sum(dim=-1)  # [B,M,N]
        #     print("row_sum stats (model out vrp_logits):", row_sum.min().item(), row_sum.mean().item(), row_sum.max().item())
        #     print("probs stats (model out vrp_logits):", vrp_logits.min().item(), vrp_logits.mean().item(), vrp_logits.max().item())
        #     row_sum_edge_probs = edge_probs.sum(dim=-1)  # [B,M,N]
        #     print("row_sum stats (model out edge_probs):", row_sum_edge_probs.min().item(), row_sum_edge_probs.mean().item(), row_sum_edge_probs.max().item())
        #     print("probs stats (model out edge_probs):", edge_probs.min().item(), edge_probs.mean().item(), edge_probs.max().item())
        # print('model called')
        # ---------------- TIMINGS: forward end (added) ----------------
        _ema_update("fwd", time.time() - t_fwd0)
        t_loss0 = time.time()
        # --------------------------------------------------------------

        # --- what do we need this batch? ---
        do_diag = (epoch > 1) and ((batch_idx == 0) or (batch_idx == steps_per_epoch - 1))
        # do_rl = opts.use_decode_rl and (batch_idx % opts.decode_rl_every == 0)
        do_rl = opts.use_decode_rl and (do_diag or (torch.rand(()) < opts.decode_rl_prob))
        need_base = bool(opts.perm_inv_loss)
        need_dist = bool(opts.use_distill and distill_union is not None)
        need_reg = bool(opts.with_regret)

        # --- cache for computed artifacts ---
        cache = {"edge_probs": edge_probs, "vrp_logits": vrp_logits,
                 "perm_m": None, "next_of": None, "loads": None,
                 "routes": None}

        def get_edge_probs():
            """Return [B,M,N,N] probabilities (cached)."""
            if cache["edge_probs"] is None:
                cache["edge_probs"] = torch.sigmoid(vrp_logits)  # fallback if model doesn't return probs
            return cache["edge_probs"]

        def get_edge_logits():
            """Return [B,M,N,N] probabilities (cached)."""
            # if cache["vrp_logits"] is None:
            #     cache["edge_probs"] = torch.sigmoid(vrp_logits)  # fallback if model doesn't return probs
            return cache["vrp_logits"]

        def get_decode():
            """Return (next_of [B,M,N], loads [B,M], perm_m list) (cached)."""
            routes = None
            if cache["next_of"] is None or cache["loads"] is None or cache["perm_m"] is None or cache["routes"] is None:
                m = fleet_b.size(1)
                perm_m = list(range(m))
                random.shuffle(perm_m)
                next_of, loads, _ = load_estimate_capacity(
                    get_edge_probs(),  # decode from probs
                    dem_b.squeeze(1),
                    perm_m,
                    capacity=1.0
                )
                cache["perm_m"], cache["next_of"], cache["loads"], cache["routes"] = perm_m, next_of, loads, routes
            return cache["next_of"], cache["loads"], cache["perm_m"]

        # ----- Base loss -----
        loss = torch.zeros((), device=device_for_loss)
        if need_base:
            _, loads, _ = get_decode()  # computed once if needed
            # P = get_edge_probs().to(device_for_loss)
            if torch.cuda.device_count() > 1:
                loads = loads.to(device_for_loss)
            # with torch.no_grad():
            #     row_sum = P.sum(dim=-1)  # [B,M,N]
            #     print("row_sum stats for P:", row_sum.min().item(), row_sum.mean().item(), row_sum.max().item())
            #     print("probs stats for P:", P.min().item(), P.mean().item(), P.max().item())

            loss = loss_f(vrp_logits, loads, targets.to(device_for_loss), target_loads.to(device_for_loss))

            # F-PIN-C auxiliary: encoder-level fleet-count supervision.
            # No-op when vcount_aux_head=False (returns None).
            aux_w = float(getattr(opts, "vcount_aux_w", 0.0) or 0.0)
            if aux_w > 0.0:
                aux = model.get_aux_loss(targets.to(device_for_loss)) if hasattr(model, "get_aux_loss") else None
                if aux is not None:
                    loss = loss + aux_w * aux

        # ---------------- TIMINGS: loss end, backward start (added) ----------------
        _ema_update("loss", time.time() - t_loss0)
        t_bwd0 = time.time()
        # -------------------------------------------------------------------------

        # ---------------- LOGGING -----------------------------------------
        # ---- epoch accounting ----
        B_cur = vrp_logits.size(0)
        running["loss_sum"] += float(loss.item()) * B_cur
        running["count"] += B_cur

        loss.backward()

        optimizer.step()

        # ---------------- TIMINGS: backward end + total (added) ----------------
        _ema_update("bwd", time.time() - t_bwd0)
        t_batch_end = time.time()
        _ema_update("tot", t_batch_end - t_batch_start)
        # ----------------------------------------------------------------------

        tr_loss_batches.append(loss.item())
        # ---- TensorBoard per-step ----
        if writer is not None and steps_per_epoch is not None and epoch is not None:
            global_step = epoch * steps_per_epoch + batch_idx
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            if edge_base is not None:
                writer.add_scalar("Train/EdgeBase", float(edge_base), global_step)
            if edge_rank is not None:
                writer.add_scalar("Train/EdgeRank", float(edge_rank), global_step)

            if decode_rl_loss is not None:
                writer.add_scalar("Train/RL_Loss", float(decode_rl_loss.item()), global_step)
            if decode_rl_loss_w is not None:
                writer.add_scalar("Train/RL_Loss_Weighted", float(decode_rl_loss_w.item()), global_step)

            if impr_mean is not None:
                writer.add_scalar("Train/RL_ImprMean", impr_mean, global_step)
            if impr_frac is not None:
                writer.add_scalar("Train/RL_ImprFrac", impr_frac, global_step)

        batch_idx += 1

        if batch_idx % opts.log_every == 0:
            print(f"Batch {batch_idx}: loss = {loss.item():.4f}, time={time.time() - t0:.2f}s")
            # ---------------- TIMINGS PRINT (added) ----------------
            print(f"[time-ema] data={t_ema['data']:.3f}s move={t_ema['move']:.3f}s "
                  f"fwd={t_ema['fwd']:.3f}s loss={t_ema['loss']:.3f}s bwd={t_ema['bwd']:.3f}s "
                  f"tot={t_ema['tot']:.3f}s")
            # ------------------------------------------------------

        t_prev_end = t_batch_end  # for next batch's data-wait timing

    print(f"Epoch {epoch} time: {time.time() - time_st_ep:.1f}s")
    print(f"[time-ema-final] data={t_ema['data']:.3f}s move={t_ema['move']:.3f}s "
          f"fwd={t_ema['fwd']:.3f}s loss={t_ema['loss']:.3f}s bwd={t_ema['bwd']:.3f}s "
          f"tot={t_ema['tot']:.3f}s")
    # ---- TensorBoard epoch summary ----
    if writer is not None and steps_per_epoch is not None and epoch is not None:
        global_step = (epoch + 1) * steps_per_epoch - 1
        train_loss_mean = running["loss_sum"] / max(1.0, running["count"])
        writer.add_scalar("Train/Loss_epoch", train_loss_mean, global_step)

    return tr_loss_batches, capa_v_sums, capa_v_means


def validate_epoch(
        val_generator: Union[DataLoader, List["CVRPInstance"]],
        model,
        loss_f,
        device,
        graph_size,
        opts,
        writer=None,
        epoch=None,
        steps_per_epoch=None,
        distill_union: Optional[Dict[int, torch.Tensor]] = None,
):
    device_for_loss = 'cuda:1' if torch.cuda.device_count() > 1 else device
    viol_b_mean_v, valid_loss_batches = [], []
    lambda_distill = getattr(opts, "lambda_distill", 0.03)
    ema = None
    ema_beta = 0.9  # 0.95 if you want smoother
    model.eval()

    # Running sums (weighted by batch size B) for epoch means
    running = dict(
        total=0.0,
        perm_inv=0.0,
        regret=0.0,
        distill=0.0,
        ap=0.0,
        p_at_100=0.0,
        count=0,
        deg=0.0,
        ent=0.0,
    )

    # Optional: only decode in validation if you truly need capacity-violation stats
    do_decode_val = bool(getattr(opts, "val_decode", False) or opts.perm_inv_loss)

    do_epoch_diag = bool(getattr(opts, "val_diag_every", 0)) and (epoch is not None) and (
            epoch % opts.val_diag_every == 0)

    last_diag_payload = None  # will hold (edge_probs, seeds, loads, targets, target_loads)
    batch_idx = 0

    running.setdefault("decoded_obj_sum", 0.0)
    running.setdefault("decoded_obj_cnt", 0.0)
    running.setdefault("k_used_sum", 0.0)
    running.setdefault("k_used_cnt", 0.0)
    running.setdefault("feas_sum", 0.0)
    running.setdefault("feas_cnt", 0.0)

    with torch.set_grad_enabled(False):
        for batch, targets_v, target_loads_v, val_id in val_generator:
            do_rl = opts.use_decode_rl and (batch_idx % opts.decode_rl_every == 0)

            loss_v = torch.zeros((), device=device_for_loss)

            targets_v = prepare_targets(targets_v, graph_size, batch, device)

            if torch.cuda.device_count() > 1:
                fleet_b, depot_b, custom_b, dem_b, dists_b = move_to(batch, device='cuda:0')
            else:
                fleet_b, depot_b, custom_b, dem_b, dists_b = move_to(batch, device)
            # print('dists_b[0][0]',dists_b[0][0])
            # --- forward ---
            vrp_logits_v, edge_probs_v = model(
                depot_b, custom_b, fleet_b, dem_b, dists_b,
                sample=False, training=False
            )

            B = edge_probs_v.size(0)

            # --- optionally decode to get loads (only if needed) ---
            need_decode_this_batch = do_decode_val or do_epoch_diag
            vrp_loads_v = None
            next_of_v = None
            if need_decode_this_batch:
                m = fleet_b.size(1)
                perm_m = list(range(m))
                # keep deterministic if you want: don't shuffle in val
                # random.shuffle(perm_m)

                next_of_v, vrp_loads_v, _ = load_estimate_capacity(
                    edge_probs_v,  # [B,M,N,N]
                    dem_b.squeeze(1),  # [B,N]
                    perm_m,
                    capacity=1.0
                )

            if do_decode_val:
                capa_violation_sum, capa_violation_mean = log_capa_violation(vrp_loads_v)
                viol_b_mean_v.append(
                    torch.mean(capa_violation_mean).item()
                    if capa_violation_mean is not None else None
                )

                if torch.cuda.device_count() > 1:
                    vrp_loads_v = vrp_loads_v.to(device_for_loss)
            else:
                # keep output shape consistent
                viol_b_mean_v.append(None)

            # ===== base loss =====
            loss_v_base = torch.zeros((), device=device_for_loss)
            if opts.perm_inv_loss:
                # requires decoded vrp_loads_v
                loss_v_base = loss_f(
                    # edge_probs_v.to(device_for_loss),
                    vrp_logits_v.to(device_for_loss),
                    vrp_loads_v,
                    targets_v.to(device_for_loss),
                    target_loads_v.to(device_for_loss),
                )
                loss_v = loss_v + loss_v_base
                running["perm_inv"] += loss_v_base.item() * B

            # ===== accumulate total and count =====
            running["total"] += loss_v.item() * B
            running["count"] += B
            valid_loss_batches.append(loss_v.item())
            batch_idx += 1

            # ---- optional diag payload (compute only when requested) ----
            # if you want diag payload, reuse the same decode
            if do_epoch_diag:
                routes_v = None
                vrp_loads_v = None
                last_diag_payload = (edge_probs_v, routes_v, vrp_loads_v, targets_v, target_loads_v)

        # ----- Compute epoch means -----
        cnt = max(1, running["count"])
        val_total_mean = running["total"] / cnt
        val_perm_inv_mean = running["perm_inv"] / cnt if opts.perm_inv_loss else 0.0
        val_regret_mean = running["regret"] / cnt if opts.with_regret else 0.0
        val_distill_mean = running["distill"] / cnt if opts.use_distill else 0.0
        val_ap_mean = running["ap"] / cnt if opts.use_distill else 0.0
        val_p_at_100_mean = running["p_at_100"] / cnt if opts.use_distill else 0.0

        # ----- TensorBoard logging (once per epoch) -----
        if writer is not None and epoch is not None and steps_per_epoch is not None:
            global_step = (epoch + 1) * steps_per_epoch - 1

            writer.add_scalar("Val/Loss", val_total_mean, global_step)

            if opts.perm_inv_loss:
                writer.add_scalar("Val/Loss_PermInv", val_perm_inv_mean, global_step)

            if opts.with_regret:
                writer.add_scalar("Val/Loss_Regret", val_regret_mean, global_step)

            if opts.use_distill:
                writer.add_scalar("Val/Loss_Distill", val_distill_mean, global_step)
                writer.add_scalar("Val/PR_AUC_union", val_ap_mean, global_step)
                writer.add_scalar("Val/TopK_Precision@100", val_p_at_100_mean, global_step)

    return valid_loss_batches, viol_b_mean_v, last_diag_payload


def prepare_targets(targets, graph_size, batch, device):
    """
    Converts targets into proper sparse tensor format for both training and validation.
    Returns:
        - targets (Tensor or stacked sparse tensors)
        - target_loads (already on the correct device)
    """
    device_for_loss = 'cuda:1' if torch.cuda.device_count() > 1 else device
    if graph_size >= 100:
        targets_i, targets_v = targets[0], targets[1]
        graph_depot = graph_size + 1
        targets_out = torch.stack([
            torch.sparse_coo_tensor(i.to(device_for_loss), v.to(device_for_loss),
                                    size=(batch[0].size(1), graph_depot, graph_depot))
            for i, v in zip(targets_i, targets_v)
        ]).to(device_for_loss)
    else:
        if isinstance(targets, list):
            targets = torch.tensor(targets)
        targets_out = targets.to(device).to_sparse()
    return targets_out


def log_capa_violation(loads):
    if loads is not None:
        # Float(batch):
        violation_sum = torch.where(loads > 1.00001, loads - 1.0000, zeros(1)).sum(1)
        # mean across vehicles
        violation_mean = torch.mean(torch.where(loads > 1.00001, loads - 1.0000, zeros(1)), dim=1)
        return violation_sum, violation_mean
    else:
        return None, None


def eval_model(model: VRP_Net,
               data_rp: List,
               normalised_data: bool,
               problem_size: int,
               problem: str,
               batch_size: int,
               device: torch.device,
               data_dist: str = None,
               opts: Union[DictConfig, NamedTuple] = None,
               ) -> Tuple[Dict[str, Any], List[RPSolution]]:
    # eval mode
    # if device.type != "cpu":
    #    torch.backends.cudnn.deterministic = True
    #    torch.backends.cudnn.benchmark = False

    # preliminaries
    vrp_to_solve, n_vehicle_default, v_cost, _ = get_preliminaries(problem_size)
    logger.info(f'Starting eval run for {vrp_to_solve} with max {opts.nr_vehicles_eval} vehicles and vehicle cost {v_cost}')
    print('opts', opts)
    eval_dataset = []
    if opts.post_process:
        from fpin.utils_all.or_tools.or_tools import ParallelSolver
        policy_cfg = opts.ls_policy_cfg.copy()
        print('ls_policy_cfg', policy_cfg)
        policy_ls = ParallelSolver(
            problem="cvrp",
            solver_args=policy_cfg,
            time_limit=opts.per_instance_time_limit_ls,
            num_workers=policy_cfg.batch_size,
            search_workers=policy_cfg.search_workers
        )

    # print('normalised_data', normalised_data)
    data_kool = prep_data_fpin(data_rp, v_max=opts.nr_vehicles)
    # print('opts.nr_vehicles_eval', opts.nr_vehicles_eval)
    test_instances, all_solvable = prep_test_data(problem_size,
                                                  data_kool,
                                                  type_=data_dist,
                                                  normed_data=normalised_data,
                                                  nr_veh=opts.nr_vehicles)
    # if all_solvable:
    #     m_set = opts.nr_vehicles_eval
    # else:
    #     m_set = None
    if opts.nr_vehicles_eval > 8:
        r_indices = list(range(n_vehicle_default))
        perm_v = list(itertools.permutations(r_indices))
    else:
        r_indices = list(range(opts.nr_vehicles_eval))
        perm_v = list(itertools.permutations(r_indices))

    model.eval()

    ### Define a range of lists to store results and metrics ###
    traveled_dists_scld_all, traveled_dists_orig_all, Dist_improved_orig, times = [], [], [], []
    costs_greedy, cost_v_greedy, routes_greedy, n_routes_greedy, sols_greedy = [], [], [], [], []
    costs, cost_v, sols, routes_all, n_routes_all = [], [], [], [], []
    sols_post, costs_post, running_sols_post, sols_hgs = [], [], [], []
    with torch.no_grad():
        for i, test_instance in enumerate(tqdm(test_instances, disable=opts.no_progress_bar)):


            t_start = time.time()
            rp_inst = data_rp[i]  # outer i only
            vals_orig = (np.concatenate((test_instance[1][:, :3], test_instance[2]), axis=0),
                         torch.FloatTensor(test_instance[4]).to(device), test_instance[5], test_instance[6])
            entities = move_to(test_instance, device, in_train=False)
            fleet_b, depot_b, custom_b, dem_b, dists_b = entities
            logits, probs = model(depot_b, custom_b, fleet_b, dem_b, dists_b,
                                         sample=True if opts.sbs_decode else False, training=False)

            if opts.giant_tour_split:
                routes_v, costs_v, stats = decode_giant_tour_split_dp(
                    edge_logits=logits,
                    dists=dists_b,
                    demands=dem_b.squeeze(1),
                    pool="any",
                    max_nr_v_eval=opts.nr_vehicles_eval,
                    vehicle_cost=v_cost)

                routes_greedy = routes_v[0]
                cost_greedy, n_routes_greedy, _ = get_travel_costs(routes_greedy, rp_inst) # [get_travel_costs(route, rp_inst) for route in routes_greedy] # [0]
                cost_v.append(cost_greedy + (n_routes_greedy * v_cost))

            elif opts.decode_vehicle_assignment:
                if opts.decode_v_assign_type == "assign_transition":
                    # decode the model AS TRAINED: per-vehicle next-node transitions
                    # (softmax over successors), walking from the depot with capacity masking.
                    routes_v, costs_v, stats = decode_transition_following(
                            edge_logits=logits,
                            dists=dists_b,
                            demands=dem_b.squeeze(1),
                            max_nr_v_eval=opts.nr_vehicles_eval,
                    )
                elif opts.decode_v_assign_type == "assign_faithful":
                    # Y_k-faithful: order routes purely by the learned per-vehicle heatmap
                    # (no geometry term), exposing the model's raw routing quality.
                    routes_v, costs_v, stats = decode_vehicle_assignment_faithful(
                            edge_logits=logits,
                            dists=dists_b,
                            demands=dem_b.squeeze(1),
                            max_nr_v_eval=opts.nr_vehicles_eval,
                            beam_width=int(getattr(opts, "faithful_beam_width", 1)),
                            use_pooled_order=bool(getattr(opts, "faithful_pooled_order", False)),
                    )
                elif opts.decode_v_assign_type == "assign_plain":
                    routes_v, costs_v, stats = decode_vehicle_assignment_1(           #     edge_logits=logits,
                            edge_logits=logits,
                            dists=dists_b,
                            demands=dem_b.squeeze(1),
                            max_nr_v_eval=opts.nr_vehicles_eval,
                    )
                else:
                    routes_v, costs_v, stats = decode_vehicle_assignment_v2(           #     edge_logits=logits,
                            edge_logits=logits,
                            dists=dists_b,
                            demands=dem_b.squeeze(1),
                            max_nr_v_eval=opts.nr_vehicles_eval,
                    )
                routes_greedy = routes_v[0]
                cost_greedy, n_routes_greedy, _ = get_travel_costs(routes_greedy, rp_inst) # [get_travel_costs(route, rp_inst) for route in routes_greedy] # [0]
                cost_v.append(cost_greedy + (n_routes_greedy * v_cost))


            # GREEDY DECODING of distribution with REPAIR
            elif opts.repair_greedy:
                sol, routes_greedy, n_routes_greedy, missing_, cost_greedy, cost_v = make_valid_and_adjust(opts,
                                                                                                       opts.nr_vehicles_eval,
                                                                                                       perm_v, probs[0],
                                                                                                       dem_b[0], v_cost,
                                                                                                       vals_orig,
                                                                                                       rp_inst)

            routes = routes_greedy

            t_constr = time.time() - t_start
            routes_greedy = [t.tolist() if not isinstance(t, list) else t for t in routes_greedy]
            costs_greedy.append(cost_greedy)
            sols_greedy.append([route for route in routes_greedy if route != [0,0]])

            # POSTPROCESSING or-tools (optional)
            if opts.post_process:
                print(f'Time for construction solution {i} before postprocess: {float(np.mean(t_constr))}')
                sols_search = policy_ls.solve([rp_inst],
                                              normed_demands=True,
                                              init_solution=routes if opts.sbs_decode else [routes],
                                              distribution=opts.coords_dist,
                                              time_construct=float(np.mean(t_constr)),
                                              grid_size=1)
                sols_post.append(sols_search[0].solution)
                costs_post.append(sols_search[0].cost)
                running_sols_post.append(sols_search[0].running_sols)
                sols = sols_post
            else:
                sols = sols_greedy if not sols_hgs else sols_hgs
            t = time.time() - t_start
            t_per_inst = t / batch_size
            times.append([t_per_inst] * batch_size)

    times = list(itertools.chain.from_iterable(times))

    return {}, make_RPSolution(problem, sols, costs_greedy, times, data_rp, running_sols_post)



def check_solution(routes, N):
    seen = []
    # print('N-1', N-1)
    for r in routes:
        for n in r:
            if n != 0:
                seen.append(n)
            # print('seen:', seen)
    seen_set = set(seen)
    seen_sorted = sorted(seen)
    # print('seen_sorted', seen_sorted)
    # print('seen_set:  ', seen_set)
    ok_all = (len(seen) == len(seen_set)) and (len(seen_set) == N-1)
    return ok_all, len(seen), len(seen_set)


def flatten_customers(routes):
    return [n for r in routes for n in r if n != 0]


import torch

def _ensure_list(x):
    # make sure we can index and iterate in python
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x

def split_orders_to_routes(
    orders, dists, dem, coords,
    cap: float,
    vehicle_cost: float,
    greedy_split_from_order_fn,
):
    """
    orders: list length B, each is a permutation (giant tour order)
    Returns:
      routes: list[B][m][t] with [0,...,0] tours
      loads:  torch [B,m] (float32) demand loads
    """
    B = len(orders)
    device = dem.device
    dem_ = dem
    if dem_.dim() == 3 and dem_.size(1) == 1:
        dem_ = dem_.squeeze(1)
    elif dem_.dim() == 3 and dem_.size(-1) == 1:
        dem_ = dem_.squeeze(-1)
    # dem_: [B,N]

    routes_out = []
    loads_out = []

    for b in range(B):
        order_b = _ensure_list(orders[b])
        # IMPORTANT: this assumes your greedy_split_from_order returns (routes, loads, k_used or similar).
        # If it returns only routes, we compute loads below.
        res = greedy_split_from_order_fn(
            order_b,
            dists[b], dem_[b], cap=cap,
            vehicle_cost=vehicle_cost, coords=coords[b],
        )
        #     order,
        #     dists,
        #     dem,
        #     cap=1.0,
        #     vehicle_cost=0.0,
        #     coords=None,
        # ):

        # handle common return signatures
        if isinstance(res, tuple):
            # out from greedy_split: obj, routes, len(routes)
            routes_b = res[1]
        else:
            routes_b = res

        # normalize to python lists
        # print('routes_b', routes_b)
        routes_b = [ _ensure_list(r) for r in routes_b ]
        routes_out.append(routes_b)

        # compute loads from routes (ignore depot=0)
        loads_b = []
        for r in routes_b:
            load = 0.0
            for node in r:
                if node != 0:
                    load += float(dem_[b, int(node)].item())
            loads_b.append(load)
        loads_out.append(loads_b)

    # pad loads to rectangular [B,m] if m varies (shouldn't, but safe)
    m_max = max(len(x) for x in loads_out) if B > 0 else 0
    loads = torch.zeros((B, m_max), device=device, dtype=torch.float32)
    for b in range(B):
        lb = loads_out[b]
        loads[b, :len(lb)] = torch.tensor(lb, device=device, dtype=torch.float32)
    return routes_out, loads
