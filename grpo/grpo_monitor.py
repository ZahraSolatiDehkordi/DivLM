import glob
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import torch
from transformers import TrainerCallback


@dataclass
class CollapseMonitorConfig:
    enabled: bool = True
    state_file: str = "collapse_monitor_state.json"
    exit_code: int = 42
    kl_hard_limit: float = 5.0
    kl_soft_limit: float = 1.5
    reward_drop_from_best: float = 0.15
    reward_crash_drop: float = 0.60
    consecutive_bad_logs: int = 2
    stable_confirm_steps: int = 400
    min_stable_step: int = 1000
    rollback_margin_steps: int = 300
    history_limit: int = 200
    max_restarts: int = 2
    accept_after_fraction: float = 0.8

    def as_dict(self):
        return asdict(self)


class CollapseMonitorTriggered(RuntimeError):
    pass


class CollapseMonitorFinalize(RuntimeError):
    def __init__(self, message, checkpoint_path):
        super().__init__(message)
        self.checkpoint_path = checkpoint_path


class CollapseMonitorFailed(RuntimeError):
    pass


def checkpoint_step(path):
    match = re.search(r"checkpoint-(\d+)$", os.path.basename(path.rstrip(os.sep)))
    return int(match.group(1)) if match else None


def find_checkpoints(adapter_dir):
    checkpoint_candidates = []
    for checkpoint_path in glob.glob(os.path.join(adapter_dir, "checkpoint-*")):
        step = checkpoint_step(checkpoint_path)
        if step is not None:
            checkpoint_candidates.append((step, checkpoint_path))
    return sorted(checkpoint_candidates, key=lambda item: item[0])


def load_monitor_state(adapter_dir, config, is_main_process=True):
    state_path = os.path.join(adapter_dir, config.state_file)
    if not os.path.exists(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        if is_main_process:
            print(f"[CollapseMonitor] Could not load resume state: {exc}")
        return {}


def resolve_resume_checkpoint(adapter_dir, config, is_main_process=True):
    checkpoint_candidates = find_checkpoints(adapter_dir)
    resume_step, resume_from = checkpoint_candidates[-1] if checkpoint_candidates else (None, None)
    resume_reason = "numerically latest checkpoint"

    monitor_state = load_monitor_state(adapter_dir, config, is_main_process=is_main_process)
    if config.enabled and monitor_state.get("status") == "collapse_detected":
        rollback_path = monitor_state.get("resume_from_checkpoint") or monitor_state.get("rollback_checkpoint")
        rollback_step = checkpoint_step(rollback_path) if rollback_path else None
        if rollback_path and rollback_step is not None and os.path.isdir(rollback_path):
            resume_step, resume_from = rollback_step, rollback_path
            resume_reason = "collapse monitor rollback checkpoint"
        elif is_main_process:
            print(
                "[CollapseMonitor] Collapse state exists, but rollback checkpoint "
                f"is unavailable: {rollback_path}. Falling back to latest checkpoint."
            )

    found_steps = [step for step, _ in checkpoint_candidates]
    return resume_step, resume_from, resume_reason, found_steps


def create_monitor_callback(adapter_dir, config, is_main_process, reward_history):
    return CollapseMonitorCallback(
        adapter_dir=adapter_dir,
        config=config,
        enabled=config.enabled and is_main_process,
        is_main_process=is_main_process,
        reward_history=reward_history,
    )


def run_training_with_monitor(
    trainer,
    resume_from,
    monitor_callback,
    config,
    is_main_process=True,
    finish_on_exit=None,
    finalize_on_collapse=None,
):
    try:
        trainer.train(resume_from_checkpoint=resume_from)
        if config.enabled:
            monitor_callback.mark_completed()
    except CollapseMonitorFinalize as exc:
        if is_main_process:
            print(str(exc))
            if finalize_on_collapse is not None:
                finalize_on_collapse(exc.checkpoint_path)
            if finish_on_exit is not None:
                finish_on_exit(0)
        sys.exit(0)
    except CollapseMonitorFailed as exc:
        if is_main_process:
            print(str(exc))
            if finish_on_exit is not None:
                finish_on_exit(1)
        sys.exit(1)
    except CollapseMonitorTriggered as exc:
        if is_main_process:
            print(str(exc))
            print(f"Exiting with code {config.exit_code} for automatic relaunch wrappers.")
            if finish_on_exit is not None:
                finish_on_exit(config.exit_code)
        sys.exit(config.exit_code)


class CollapseMonitorCallback(TrainerCallback):
    REWARD_KEYS = (
        "rewards/reward_func/mean",
        "train/rewards/reward_func/mean",
        "train/reward",
        "train/rewards/mean",
        "reward",
        "reward_mean",
        "train/total_reward",
    )
    KL_KEYS = ("kl", "train/kl", "objective/kl", "train/objective/kl")

    def __init__(
        self,
        adapter_dir,
        config,
        enabled=True,
        is_main_process=True,
        reward_history=None,
    ):
        self.adapter_dir = adapter_dir
        self.state_path = os.path.join(adapter_dir, config.state_file)
        self.enabled = enabled
        self.is_main_process = is_main_process
        self.kl_hard_limit = config.kl_hard_limit
        self.kl_soft_limit = config.kl_soft_limit
        self.reward_drop_from_best = config.reward_drop_from_best
        self.reward_crash_drop = config.reward_crash_drop
        self.consecutive_bad_logs = config.consecutive_bad_logs
        self.stable_confirm_steps = config.stable_confirm_steps
        self.min_stable_step = config.min_stable_step
        self.rollback_margin_steps = config.rollback_margin_steps
        self.history_limit = config.history_limit
        self.max_restarts = config.max_restarts
        self.accept_after_fraction = config.accept_after_fraction
        self.reward_history = reward_history if reward_history is not None else []

        self.candidate_checkpoints = []
        self.stable_checkpoints = []
        self.last_confirmed_stable_checkpoint = None
        self.history = []
        self.best_reward = None
        self.bad_log_count = 0
        self.last_unhealthy_step = None
        self.restart_count = 0

        self._load_state()
        self._scan_existing_checkpoints()

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        try:
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().item()
            elif isinstance(value, np.ndarray):
                value = float(np.mean(value))
            else:
                value = float(value)
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        except (TypeError, ValueError):
            return None

    def _metric(self, logs, keys):
        for key in keys:
            if key in logs:
                value = self._to_float(logs.get(key))
                if value is not None:
                    return value
        return None

    def _recent_reward_from_history(self):
        rewards = self.reward_history
        if not rewards:
            return None
        return self._to_float(np.mean(rewards[-100:]))

    def _load_state(self):
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            if self.is_main_process:
                print(f"[CollapseMonitor] Could not read state file: {exc}")
            return

        self.candidate_checkpoints = data.get("candidate_checkpoints", []) or []
        self.stable_checkpoints = data.get("stable_checkpoints", []) or []
        self.last_confirmed_stable_checkpoint = data.get("last_confirmed_stable_checkpoint")
        self.history = data.get("history", [])[-self.history_limit:]
        self.best_reward = self._to_float(data.get("best_reward"))
        self.bad_log_count = int(data.get("bad_log_count", 0) or 0)
        self.last_unhealthy_step = data.get("last_unhealthy_step")
        self.restart_count = int(data.get("restart_count", 0) or 0)

        if data.get("status") in {"collapse_detected", "rollback_resume_selected"}:
            self.best_reward = None
            self.bad_log_count = 0
            self.last_unhealthy_step = None

    def _scan_existing_checkpoints(self):
        for checkpoint_path in glob.glob(os.path.join(self.adapter_dir, "checkpoint-*")):
            step = checkpoint_step(checkpoint_path)
            if step is not None:
                self._register_checkpoint(step, checkpoint_path)

    def _register_checkpoint(self, step, path):
        path = os.path.abspath(path)
        known_steps = {item.get("step") for item in self.candidate_checkpoints}
        if step not in known_steps:
            self.candidate_checkpoints.append({"step": int(step), "path": path})
            self.candidate_checkpoints = sorted(
                self.candidate_checkpoints, key=lambda item: item["step"]
            )

    def _confirm_stable_checkpoints(self, current_step):
        stable_steps = {item.get("step") for item in self.stable_checkpoints}
        for item in self.candidate_checkpoints:
            step = item["step"]
            if step in stable_steps:
                continue
            if step < self.min_stable_step:
                continue
            if current_step - step < self.stable_confirm_steps:
                continue
            if self.last_unhealthy_step is not None and self.last_unhealthy_step >= step:
                continue
            if not os.path.isdir(item["path"]):
                continue

            stable_item = {
                "step": int(step),
                "path": item["path"],
                "confirmed_at_step": int(current_step),
            }
            self.stable_checkpoints.append(stable_item)
            self.stable_checkpoints = sorted(
                self.stable_checkpoints, key=lambda entry: entry["step"]
            )
            self.last_confirmed_stable_checkpoint = stable_item

    def _choose_rollback_checkpoint(self, current_step):
        min_step = current_step - self.rollback_margin_steps
        stable_existing = [
            item for item in self.stable_checkpoints
            if item.get("step", -1) <= min_step and os.path.isdir(item.get("path", ""))
        ]
        if stable_existing:
            return max(stable_existing, key=lambda item: item["step"])

        stable_existing = [
            item for item in self.stable_checkpoints
            if item.get("step", -1) < current_step and os.path.isdir(item.get("path", ""))
        ]
        if stable_existing:
            return max(stable_existing, key=lambda item: item["step"])

        candidate_existing = [
            item for item in self.candidate_checkpoints
            if item.get("step", -1) <= min_step and os.path.isdir(item.get("path", ""))
        ]
        if candidate_existing:
            return max(candidate_existing, key=lambda item: item["step"])

        candidate_existing = [
            item for item in self.candidate_checkpoints
            if item.get("step", -1) < current_step and os.path.isdir(item.get("path", ""))
        ]
        return max(candidate_existing, key=lambda item: item["step"]) if candidate_existing else None

    def _write_state(self, status="running", extra=None):
        if not self.is_main_process:
            return
        os.makedirs(self.adapter_dir, exist_ok=True)
        payload = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "state_file": self.state_path,
            "best_reward": self.best_reward,
            "bad_log_count": self.bad_log_count,
            "last_unhealthy_step": self.last_unhealthy_step,
            "restart_count": self.restart_count,
            "last_confirmed_stable_checkpoint": self.last_confirmed_stable_checkpoint,
            "candidate_checkpoints": self.candidate_checkpoints[-100:],
            "stable_checkpoints": self.stable_checkpoints[-100:],
            "history": self.history[-self.history_limit:],
            "settings": {
                "kl_hard_limit": self.kl_hard_limit,
                "kl_soft_limit": self.kl_soft_limit,
                "reward_drop_from_best": self.reward_drop_from_best,
                "reward_crash_drop": self.reward_crash_drop,
                "consecutive_bad_logs": self.consecutive_bad_logs,
                "stable_confirm_steps": self.stable_confirm_steps,
                "min_stable_step": self.min_stable_step,
                "rollback_margin_steps": self.rollback_margin_steps,
                "max_restarts": self.max_restarts,
                "accept_after_fraction": self.accept_after_fraction,
            },
        }
        if extra:
            payload.update(extra)
        tmp_path = self.state_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, self.state_path)

    def mark_completed(self):
        self._write_state(status="completed")

    def on_save(self, args, state, control, **kwargs):
        if not self.enabled:
            return control
        step = int(state.global_step or 0)
        if step <= 0:
            return control
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{step}")
        self._register_checkpoint(step, checkpoint_path)
        self._confirm_stable_checkpoints(step)
        self._write_state(status="running")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.enabled:
            return control
        logs = logs or {}
        step = int(state.global_step or 0)
        if step <= 0:
            return control

        kl = self._metric(logs, self.KL_KEYS)
        reward = self._metric(logs, self.REWARD_KEYS)
        if reward is None:
            reward = self._recent_reward_from_history()

        reward_drop = 0.0
        if reward is not None and self.best_reward is not None:
            reward_drop = max(0.0, self.best_reward - reward)

        hard_kl = kl is not None and kl >= self.kl_hard_limit
        soft_kl = kl is not None and kl >= self.kl_soft_limit
        reward_drop_bad = reward_drop >= self.reward_drop_from_best
        reward_crash = reward_drop >= self.reward_crash_drop
        unhealthy = hard_kl or reward_crash or (soft_kl and reward_drop_bad)

        reason = {
            "kl": kl,
            "reward": reward,
            "best_reward": self.best_reward,
            "reward_drop_from_best": reward_drop,
            "hard_kl": hard_kl,
            "soft_kl": soft_kl,
            "reward_drop_bad": reward_drop_bad,
            "reward_crash": reward_crash,
        }

        if unhealthy:
            self.bad_log_count += 1
            self.last_unhealthy_step = step
        else:
            self.bad_log_count = max(0, self.bad_log_count - 1)
            if reward is not None and (self.best_reward is None or reward > self.best_reward):
                self.best_reward = reward

        self._confirm_stable_checkpoints(step)

        entry = {
            "step": step,
            "kl": kl,
            "reward": reward,
            "best_reward": self.best_reward,
            "reward_drop_from_best": reward_drop,
            "bad_log_count": self.bad_log_count,
            "unhealthy": unhealthy,
        }
        self.history.append(entry)
        self.history = self.history[-self.history_limit:]

        collapse_confirmed = (
            step >= self.min_stable_step
            and (
                hard_kl
                or (soft_kl and self.bad_log_count >= self.consecutive_bad_logs)
            )
        )

        if collapse_confirmed:
            rollback = self._choose_rollback_checkpoint(step)
            rollback_path = rollback.get("path") if rollback else None
            max_steps = int(getattr(state, "max_steps", 0) or 0)
            progress_fraction = (float(step) / float(max_steps)) if max_steps > 0 else None
            accept_due_to_progress = (
                progress_fraction is not None
                and progress_fraction >= self.accept_after_fraction
            )
            restart_budget_exhausted = self.restart_count >= self.max_restarts
            should_finalize = rollback_path and (
                accept_due_to_progress or restart_budget_exhausted
            )
            extra = {
                "collapse_step": step,
                "collapse_reason": reason,
                "rollback_checkpoint": rollback_path,
                "resume_from_checkpoint": rollback_path,
                "max_steps": max_steps,
                "progress_fraction": progress_fraction,
                "accept_due_to_progress": accept_due_to_progress,
                "restart_budget_exhausted": restart_budget_exhausted,
            }

            if not rollback_path:
                self._write_state(status="collapse_no_checkpoint", extra=extra)
                message = (
                    "[CollapseMonitor] Collapse detected at step "
                    f"{step}, but no checkpoint was available for rollback."
                )
                if self.is_main_process:
                    print("\n" + "=" * 80)
                    print(message)
                    print(f"Reason: {reason}")
                    print(f"State written to: {self.state_path}")
                    print("=" * 80 + "\n")
                raise CollapseMonitorFailed(message)

            if should_finalize:
                extra["final_checkpoint"] = rollback_path
                self._write_state(status="finalized_after_collapse", extra=extra)
                message = (
                    "[CollapseMonitor] Collapse detected at step "
                    f"{step}. Finalizing checkpoint: {rollback_path}"
                )
                if self.is_main_process:
                    print("\n" + "=" * 80)
                    print(message)
                    print(f"Reason: {reason}")
                    print(f"Progress fraction: {progress_fraction}")
                    print(f"Restart count: {self.restart_count}/{self.max_restarts}")
                    print(f"State written to: {self.state_path}")
                    print("=" * 80 + "\n")
                raise CollapseMonitorFinalize(message, rollback_path)

            self.restart_count += 1
            extra["restart_count"] = self.restart_count
            self._write_state(status="collapse_detected", extra=extra)

            message = (
                "[CollapseMonitor] Collapse detected at step "
                f"{step}. Rollback checkpoint: {rollback_path or 'NONE'}"
            )
            if self.is_main_process:
                print("\n" + "=" * 80)
                print(message)
                print(f"Reason: {reason}")
                print(f"Restart count: {self.restart_count}/{self.max_restarts}")
                print(f"State written to: {self.state_path}")
                print("Relaunch the same command to resume from the rollback checkpoint.")
                print("=" * 80 + "\n")
            raise CollapseMonitorTriggered(message)

        self._write_state(status="running")
        return control
