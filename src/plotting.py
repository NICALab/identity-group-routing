from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save_figure(fig, out_png: Path) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight")


def plot_confusion_matrix(cm: np.ndarray, out_png: Path, *, title: str, normalize: bool = True) -> None:
    cm_arr = np.asarray(cm, dtype=np.float32)
    if cm_arr.ndim != 2:
        raise ValueError(f"Expected 2D confusion matrix, got {cm_arr.shape}")
    plot_arr = cm_arr.copy()
    if normalize:
        row_sum = plot_arr.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum <= 0.0, 1.0, row_sum)
        plot_arr = plot_arr / row_sum

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(plot_arr, cmap="Blues", aspect="auto")
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save_figure(fig, out_png)
    plt.close(fig)


def plot_training_history(
    train_losses: list[float],
    val_scores: list[float] | None,
    out_png: Path,
    *,
    title: str,
    score_label: str,
    best_epoch: int | None = None,
    save_svg: bool = True,
) -> None:
    if not train_losses:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color="#C05621", lw=1.6, label="train loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if val_scores is not None and len(val_scores) == len(train_losses):
        ax2 = ax.twinx()
        ax2.plot(epochs, val_scores, color="#1D4ED8", lw=1.6, label=score_label)
        ax2.set_ylabel(score_label, color="#1D4ED8")
        ax2.tick_params(axis="y", labelcolor="#1D4ED8")
        if best_epoch is not None and best_epoch >= 1:
            ax.axvline(int(best_epoch), color="#111827", lw=1.0, ls="--", alpha=0.65)
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
    elif best_epoch is not None and best_epoch >= 1:
        ax.axvline(int(best_epoch), color="#111827", lw=1.0, ls="--", alpha=0.65)
    if handles:
        ax.legend(handles, labels, loc="best", frameon=False)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    if bool(save_svg):
        fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_behavior_predictions(y_true: np.ndarray, y_pred: np.ndarray, out_png: Path, *, title: str) -> None:
    t = np.arange(len(y_true), dtype=np.int64)
    fig, ax = plt.subplots(figsize=(12.5, 3.8))
    ax.plot(t, y_true, color="#1F2937", lw=1.0, label="target")
    ax.plot(t, y_pred, color="#DC2626", lw=1.0, alpha=0.82, label="prediction")
    ax.set_xlabel("time")
    ax.set_ylabel("motor")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    _save_figure(fig, out_png)
    plt.close(fig)


def plot_latent_3d_time(
    latent_td: np.ndarray,
    out_png: Path,
    title: str,
    subset_latents: list[np.ndarray] | None = None,
    variant_name: str | None = None,
    color_values: np.ndarray | None = None,
    color_label: str = "time",
    cmap: str = "viridis",
) -> None:
    if latent_td.ndim != 2 or latent_td.shape[1] < 3:
        return

    xyz_ref = np.asarray(latent_td[:, :3], dtype=np.float64)
    mu_ref = xyz_ref.mean(axis=0, keepdims=True)
    x0 = xyz_ref - mu_ref
    try:
        _, _, v_t = np.linalg.svd(x0, full_matrices=False)
        rot = v_t.T.astype(np.float64, copy=False)
    except np.linalg.LinAlgError:
        rot = np.eye(3, dtype=np.float64)

    def _rotate_xyz(z_td: np.ndarray) -> np.ndarray:
        xyz = np.asarray(z_td[:, :3], dtype=np.float64)
        return ((xyz - mu_ref) @ rot).astype(np.float32, copy=False)

    color_arr = None if color_values is None else np.asarray(color_values, dtype=np.float32).reshape(-1)

    def _panel_colors(n_t: int) -> np.ndarray:
        if color_arr is None or color_arr.shape[0] != n_t:
            return np.arange(n_t, dtype=np.float32)
        return color_arr

    def _draw_panel(ax, z_td: np.ndarray, panel_title: str, show_axis_labels: bool = True):
        xyz = _rotate_xyz(z_td)
        cvals = _panel_colors(xyz.shape[0])
        sc_loc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=cvals, cmap=cmap, s=4, alpha=0.70)
        ax.set_xlabel("r1" if show_axis_labels else "")
        ax.set_ylabel("r2" if show_axis_labels else "")
        ax.set_zlabel("r3" if show_axis_labels else "")
        ax.set_title(panel_title, fontsize=10)
        ax.view_init(elev=24, azim=38)
        return sc_loc

    has_subsets = subset_latents is not None and len(subset_latents) > 0
    if not has_subsets:
        fig = plt.figure(figsize=(6.4, 5.6))
        ax = fig.add_subplot(111, projection="3d")
        sc = _draw_panel(ax, latent_td, title if not variant_name else f"{title} [{variant_name}]")
        cb = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.08)
        cb.set_label(color_label)
        fig.tight_layout()
        _save_figure(fig, out_png)
        plt.close(fig)
        return

    fig = plt.figure(figsize=(19.0, 5.2))
    axes = [fig.add_subplot(1, 4, i + 1, projection="3d") for i in range(4)]
    panel_data: list[np.ndarray | None] = [latent_td]
    for i in range(3):
        if i < len(subset_latents) and subset_latents[i].ndim == 2 and subset_latents[i].shape[1] >= 3:
            panel_data.append(subset_latents[i])
        else:
            panel_data.append(None)

    sc = _draw_panel(axes[0], panel_data[0], "Average embedding", show_axis_labels=True)
    for i in range(3):
        z_i = panel_data[i + 1]
        if z_i is not None:
            _draw_panel(axes[i + 1], z_i, f"Subset #{i + 1}", show_axis_labels=False)
        else:
            axes[i + 1].text2D(0.35, 0.5, "N/A", transform=axes[i + 1].transAxes, fontsize=12)
            axes[i + 1].set_title(f"Subset #{i + 1}", fontsize=10)
            axes[i + 1].set_axis_off()

    fig.suptitle(title if not variant_name else f"{title} [{variant_name}]")
    fig.subplots_adjust(left=0.03, right=0.90, top=0.90, bottom=0.05, wspace=0.08)
    cax = fig.add_axes([0.92, 0.17, 0.012, 0.66])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(color_label)
    _save_figure(fig, out_png)
    plt.close(fig)
