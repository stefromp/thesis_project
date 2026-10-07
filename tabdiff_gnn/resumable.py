"""
Resumable TabDiff training -- used for news ONLY (its 8000 epochs do not fit one
48 h job). Every other dataset trains with TabDiff's own Trainer, unchanged.

ResumableTrainer subclasses tabdiff.trainer.Trainer. Its run_loop is TabDiff's
Trainer.run_loop copied VERBATIM (third_party/TabDiff/tabdiff/trainer.py, lines
124-295 @ 5ecdb33) plus exactly two marked blocks
  # >>> ROUTE-A RESUME HOOK ... # <<< ROUTE-A RESUME HOOK
  1. before `start_epoch = self.curr_epoch`: restore a saved run state;
  2. at the end of each epoch's body: save the run state every `save_every`
     epochs, and pause (save + leave the loop) before the job's time budget runs
     out or at `stop_after_epoch` (tests).
tests/test_tabdiff_gnn.py proves (a) the loop minus the two blocks is
byte-identical to TabDiff's source, and (b) on CPU, a run interrupted at epoch
k and resumed gives bit-identical weights, EMA weights, schedules, optimizer /
LR-scheduler state and best-checkpoint choice as an uninterrupted run -- and
as TabDiff's own Trainer.

The run state (<run>/ckpt/resume_state.pt) holds everything that affects the
run: denoiser + learnable noise schedules, the EMA denoiser + EMA schedules,
AdamW state, ReduceLROnPlateau state, the epoch counter (=> the continuous-loss
annealing position), patience / closs_weight / dloss_weight, the best-loss and
best-EMA-loss tracking (TabDiff saves best checkpoints only after epoch 4000)
together with the exact bytes of the best_model_* / best_ema_model_* files on
disk at that epoch (restored verbatim, so a crash after a later best can not
leak into the resumed run), and the torch CPU / CUDA, numpy and python RNG
states. The DataLoader's shuffling and worker seeds are drawn from the torch
RNG at each epoch, so they are restored with it.
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import random
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TABDIFF_ROOT = os.path.join(REPO, "third_party", "TabDiff")
if TABDIFF_ROOT not in sys.path:
    sys.path.insert(0, TABDIFF_ROOT)
from tabdiff.trainer import Trainer, print_with_bar, update_ema  # noqa: E402,F401  (names the copied loop uses,
                                                                 # from the same module namespace as TabDiff's)

RESUME_FILE = "resume_state.pt"
STATE_FORMAT = 1
BEST_PATTERNS = ("best_model_*", "best_ema_model_*")


class ResumableTrainer(Trainer):
    def __init__(self, *args, resume_path, save_every=100, budget_deadline=None,
                 stop_after_epoch=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.resume_path = resume_path
        self.save_every = save_every
        self.budget_deadline = budget_deadline          # time.time() by which this job must have saved
        self.stop_after_epoch = stop_after_epoch        # tests: pause after this epoch
        self.resume_state = (torch.load(resume_path, map_location=self.device, weights_only=False)
                             if os.path.exists(resume_path) else None)
        self.resumed_from_epoch = self.resume_state["epoch"] if self.resume_state else 0
        self.paused_at = None
        self.n_saves = 0
        self._last_t = time.time()
        self._epoch_s = []

    # ---- state ----------------------------------------------------------------
    def _rng_state(self) -> dict:
        return {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(), "python": random.getstate()}

    def _save_run_state(self, epoch_done, patience, closs_weight, dloss_weight, best_loss, best_ema_loss):
        best_files = {}
        for pat in BEST_PATTERNS:
            for f in glob.glob(os.path.join(self.model_save_path, pat)):
                with open(f, "rb") as fh:
                    best_files[os.path.basename(f)] = fh.read()
        state = {
            "format": STATE_FORMAT, "epoch": epoch_done, "steps": self.steps,
            "denoise_fn": self.diffusion._denoise_fn.state_dict(),
            "num_schedule": self.diffusion.num_schedule.state_dict(),
            "cat_schedule": self.diffusion.cat_schedule.state_dict(),
            "ema_denoise_fn": self.ema_model.state_dict(),
            "ema_num_schedule": self.ema_num_schedule.state_dict(),
            "ema_cat_schedule": self.ema_cat_schedule.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict(),
            "patience": patience, "closs_weight": closs_weight, "dloss_weight": dloss_weight,
            "best_loss": best_loss, "best_ema_loss": best_ema_loss, "best_files": best_files,
            "rng": self._rng_state(),
            "saved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        tmp = self.resume_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.resume_path)
        self.n_saves += 1

    def _restore_run_state(self):
        st = self.resume_state
        if st.get("format") != STATE_FORMAT or st["steps"] != self.steps:
            raise RuntimeError(f"{self.resume_path}: format/steps {st.get('format')}/{st['steps']} "
                               f"do not match this run ({STATE_FORMAT}/{self.steps})")
        self.diffusion._denoise_fn.load_state_dict(st["denoise_fn"])
        self.diffusion.num_schedule.load_state_dict(st["num_schedule"])
        self.diffusion.cat_schedule.load_state_dict(st["cat_schedule"])
        self.ema_model.load_state_dict(st["ema_denoise_fn"])
        self.ema_num_schedule.load_state_dict(st["ema_num_schedule"])
        self.ema_cat_schedule.load_state_dict(st["ema_cat_schedule"])
        self.optimizer.load_state_dict(st["optimizer"])
        self.scheduler.load_state_dict(st["lr_scheduler"])
        for pat in BEST_PATTERNS:                       # best files = exactly those at the saved epoch
            for f in glob.glob(os.path.join(self.model_save_path, pat)):
                os.remove(f)
        for name, blob in st["best_files"].items():
            with open(os.path.join(self.model_save_path, name), "wb") as fh:
                fh.write(blob)
        self.curr_epoch = st["epoch"]
        torch.set_rng_state(st["rng"]["torch"].cpu())
        if st["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in st["rng"]["cuda"]])
        np.random.set_state(st["rng"]["numpy"])
        random.setstate(st["rng"]["python"])
        print_with_bar(f"Resumed run state of epoch {self.curr_epoch} from {self.resume_path}")
        return st["patience"], st["closs_weight"], st["dloss_weight"], st["best_loss"], st["best_ema_loss"]

    def _after_epoch(self, epoch, patience, closs_weight, dloss_weight, best_loss, best_ema_loss) -> bool:
        """True = state saved, leave the loop (pause). Never touches any RNG."""
        now = time.time()
        self._epoch_s.append(now - self._last_t)
        self._last_t = now
        done = epoch + 1
        if done >= self.steps:
            return False                                # finished: nothing to resume
        stop = self.stop_after_epoch is not None and done >= self.stop_after_epoch
        if self.budget_deadline is not None:
            recent = max(self._epoch_s[-20:])
            # the next save point must still fit: 3 slow epochs + one save of headroom
            stop = stop or now + 3 * recent + 120 > self.budget_deadline
        if stop or (self.save_every and done % self.save_every == 0):
            self._save_run_state(done, patience, closs_weight, dloss_weight, best_loss, best_ema_loss)
        if stop:
            self.paused_at = done
            print_with_bar(f"Paused after epoch {done}/{self.steps}; run state saved to {self.resume_path}")
        return stop

    # ---- TabDiff's Trainer.run_loop, verbatim + the two hooks -----------------
    def run_loop(self):
        patience = 0
        closs_weight, dloss_weight = self.c_lambda, self.d_lambda
        best_loss = np.inf
        best_ema_loss = np.inf
        best_val_loss = np.inf
        start_time = time.time()
        print_with_bar(f"Starting Trainin Loop, total number of epoch = {self.steps}")
        # Set up wandb's step metric
        self.logger.define_metric("epoch")
        self.logger.define_metric("*", step_metric="epoch")
        
        # >>> ROUTE-A RESUME HOOK (restore; not in TabDiff)
        if self.resume_state is not None:
            patience, closs_weight, dloss_weight, best_loss, best_ema_loss = self._restore_run_state()
        # <<< ROUTE-A RESUME HOOK
        start_epoch = self.curr_epoch
        if start_epoch > 0:
            print_with_bar(f"Resuming training from epoch {start_epoch}, with validation check every {self.check_val_every} epoches")
        for epoch in range (start_epoch, self.steps):
            self.curr_epoch = epoch+1
            # Set up pbar
            pbar = tqdm(self.train_iter, total=len(self.train_iter))
            pbar.set_description(f"Epoch {epoch+1}/{self.steps}")
            
            # Compute the loss weights
            if self.closs_weight_schedule == "fixed":
                pass
            elif self.closs_weight_schedule == "anneal":
                frac_done = epoch / self.steps
                closs_weight = self.c_lambda * (1 - frac_done)
            else:
                raise NotImplementedError(f"The continuous loss weight schedule {self.closs_weight_schedule} is not implemneted")

            # Training Step
            curr_dloss = 0.0
            curr_closs = 0.0
            curr_count = 0
            curr_lr = self.optimizer.param_groups[0]['lr']
            for batch in pbar:
                x = batch.float().to(self.device)
                batch_dloss, batch_closs = self._run_step(x, closs_weight, dloss_weight)
                curr_dloss += batch_dloss.item() * len(x)
                curr_closs += batch_closs.item() * len(x)
                curr_count += len(x)
                pbar.set_postfix({
                    "lr": curr_lr,
                    "DLoss": np.around(curr_dloss/curr_count, 4),
                    "CLoss": np.around(curr_closs/curr_count, 4),
                    "TotalLoss": np.around((curr_dloss + curr_closs)/curr_count, 4),
                    "closs_weight": closs_weight,
                    "dloss_weight": dloss_weight,
                })
                
            # Log training Loss
            log_dict = {}
            mloss = np.around(curr_dloss / curr_count, 4)
            gloss = np.around(curr_closs / curr_count, 4)
            total_loss = mloss + gloss
            if np.isnan(gloss):
                    print('Finding Nan in gaussian loss')
                    break
            loss_dict = {
                "epoch": epoch + 1,
                "lr": curr_lr,
                "closs_weight": closs_weight,
                "dloss_weight": dloss_weight,
                "loss/c_loss": gloss,
                "loss/d_loss": mloss,
                "loss/total_loss": total_loss
            }
            log_dict.update(loss_dict)
            
            # Log the learned noise schedules for numerical dimensions
            if self.dataset.d_numerical > 0:    # numerical data is not empty
                num_noise_dict = {}
                if self.diffusion.num_schedule.rho().dim() == 0:   # non-learnable num schedule
                    num_noise_dict = {"num_noise/rho": self.diffusion.num_schedule.rho().item()}
                else:
                    num_noise_dict = {f"num_noise/rho_col_{i}": value.item() for i, value in enumerate(self.diffusion.num_schedule.rho())}
                log_dict.update(num_noise_dict)            

            # Log the learned noise schedules for categlrical dimensions
            if len(self.dataset.categories) > 0:    # categorical data is not empty
                cat_noise_dict = {}
                if self.diffusion.cat_schedule.k().dim() == 0:   # non-learnable cat schedule
                    cat_noise_dict = {"cat_noise/k": self.diffusion.cat_schedule.k().item()}
                else:
                    cat_noise_dict = {f"cat_noise/k_col_{i}": value.item() for i, value in enumerate(self.diffusion.cat_schedule.k())}
                log_dict.update(cat_noise_dict)
            
            # Adjust learning rate
            if self.lr_scheduler == 'reduce_lr_on_plateau':
                self.scheduler.step(total_loss)
            elif  self.lr_scheduler == 'anneal':
                self._anneal_lr(epoch)
            elif self.lr_scheduler == 'fixed':
                pass
            else:
                raise NotImplementedError(f"LR scheduler with name '{self.lr_scheduler}' is not implemented")
            
            # Update EMA models
            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters(), rate=self.ema_decay)
            update_ema(self.ema_num_schedule.parameters(), self.diffusion.num_schedule.parameters(), rate=self.ema_decay)
            update_ema(self.ema_cat_schedule.parameters(), self.diffusion.cat_schedule.parameters(), rate=self.ema_decay)

            # Save ckpt base on the best training loss
            if total_loss < best_loss and self.curr_epoch > 4000:
                best_loss = total_loss
                to_remove = glob.glob(os.path.join(self.model_save_path, f"best_model_*"))
                if to_remove:
                    os.remove(to_remove[0])
                state_dicts = {
                    'denoise_fn': self.diffusion._denoise_fn.state_dict(), 
                    'num_schedule':self.diffusion.num_schedule.state_dict(), 
                    'cat_schedule': self.diffusion.cat_schedule.state_dict(),
                }
                torch.save(state_dicts, os.path.join(self.model_save_path, f'best_model_{np.round(total_loss,4)}_{epoch+1}.pt'))
                patience = 0
            else:
                patience += 1   # increment patience if best loss is not surpassed
            
            # Compute and log EMA model loss
            curr_model, curr_num_schedule, curr_cat_schedule = self.to_ema_model()
            ema_mloss, ema_gloss = self.compute_loss()
            self.to_model(curr_model, curr_num_schedule, curr_cat_schedule)
            ema_total_loss = ema_mloss + ema_gloss
            ema_loss_dict = {
                "ema_loss/c_loss": ema_gloss,
                "ema_loss/d_loss": ema_mloss,
                "ema_loss/total_loss": ema_total_loss
            }
            
            # Save the best ema ckpt
            if ema_total_loss < best_ema_loss and self.curr_epoch > 4000:
                best_ema_loss = ema_total_loss
                to_remove = glob.glob(os.path.join(self.model_save_path, f"best_ema_model_*"))
                if to_remove:
                    os.remove(to_remove[0])
                state_dicts = {
                    'denoise_fn': self.ema_model.state_dict(), 
                    'num_schedule':self.ema_num_schedule.state_dict(), 
                    'cat_schedule': self.ema_cat_schedule.state_dict(),
                }
                torch.save(state_dicts, os.path.join(self.model_save_path, f'best_ema_model_{np.round(ema_total_loss,4)}_{epoch+1}.pt'))
            
            # Evaluate Sample Quality
            if (epoch+1) % self.check_val_every == 0:
                state_dicts = {
                    'denoise_fn': self.diffusion._denoise_fn.state_dict(), 
                    'num_schedule':self.diffusion.num_schedule.state_dict(), 
                    'cat_schedule': self.diffusion.cat_schedule.state_dict(),
                }
                torch.save(state_dicts, os.path.join(self.model_save_path, f'model_{epoch+1}.pt'))
                
                print_with_bar(f"Routine Generation Evaluation every {self.check_val_every}, currently at epoch #{epoch+1}, wiht total_loss={total_loss}.")
                out_metrics, _, _ = self.evaluate_generation(save_metric_details=True, plot_density=True)
                log_dict.update(out_metrics)
                print(f"Eval Resutls of the Non-EMA model:\n {out_metrics}")

                # Evaluate the EMA model
                torch.save(self.ema_model.state_dict(), os.path.join(self.model_save_path, f'ema_model_{epoch+1}.pt'))
                ema_out_metrics, _, _ = self.evaluate_generation(ema=True, save_metric_details=True, plot_density=True)
                log_dict.update({
                    "ema": ema_out_metrics,
                })
                print(f"Eval Resutls of the EMA model:\n {ema_out_metrics}")
            
            # Submit logs
            self.logger.log(log_dict)
            # >>> ROUTE-A RESUME HOOK (save / pause; not in TabDiff)
            if self._after_epoch(epoch, patience, closs_weight, dloss_weight, best_loss, best_ema_loss):
                break
            # <<< ROUTE-A RESUME HOOK

        end_time = time.time()
        print_with_bar(f"Ending Trainnig Loop, totoal training time = {end_time - start_time}")
        self.logger.log({
            'training_time': end_time - start_time
        })
