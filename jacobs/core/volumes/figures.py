import logging
from functools import cached_property

import numpy as np
from matplotlib import pyplot as plt
from polpo.dataset.plot import plot_nested
from polpo.plot.pyplot import make_legend_handle, plot_grid, plot_layers
from polpo.workflow.task import TaskRunner, task
from sklearn.linear_model import LinearRegression

logging.getLogger("fontTools.subset").setLevel(logging.WARNING)


def plot_volumes_over_time(
    dataset,
    ax=None,
    colors=None,
    legend=False,
    ylabel="Volume",
    **kwargs,
):
    if ax is None:
        _, ax = plt.subplots()

    plot_nested(dataset, ax=ax, colors=colors, include_label=legend, **kwargs)

    ax.set_xlabel("Week")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 42)

    if legend:
        ax.legend()

    return ax


def plot_trajectory_summary(
    models, ax=None, colors=None, legend=False, zero_slope=True, **kwargs
):
    if ax is None:
        _, ax = plt.subplots()

    colors = colors or {}

    for subject, model in models.items():
        ax.scatter(
            model.intercept_,
            model.coef_[0],
            color=colors.get(subject, None),
            label=subject,
            **kwargs,
        )

    if zero_slope:
        ax.axhline(0, color="0.85", linewidth=1, zorder=0)

    ax.set_xlabel("Intercept")
    ax.set_ylabel("Slope")

    if legend:
        ax.legend()

    return ax


def fit_trajectory(sessions):
    X = np.array(list(sessions.keys())).reshape(-1, 1)
    y = np.array(list(sessions.values()))

    return LinearRegression().fit(X, y)


def pair_hemispheres(data):
    paired = {}

    for name, item in data.items():
        side, struct = name.split(" ", 1)
        side = {"Left": "left", "Right": "right"}[side]

        paired.setdefault(struct, {})[side] = item

    return paired


class VolumeFiguresRunner(TaskRunner):
    def __init__(self, figures_dir, load_volumes, colors, state_dir=None):
        super().__init__(state_dir or figures_dir / ".state")

        self.figures_dir = figures_dir
        self.colors = colors

        self._load_volumes = load_volumes

    @classmethod
    def from_volumes(cls, figures_dir, volumes, colors, **kwargs):
        return cls(figures_dir, lambda: volumes, colors, **kwargs)

    @cached_property
    def volumes(self):
        with self.timer("load_volumes"):
            volumes = self._load_volumes()

        self.logger.info(
            "Loaded data in %.1fs",
            self.timer.duration("load_volumes"),
        )
        return volumes

    @cached_property
    def log_volumes(self):
        return {
            name: grouped.map_values(lambda dataset: dataset.map_values(np.log))
            for name, grouped in self.volumes.items()
        }

    @task
    def longitudinal(self):
        self._plot_longitudinal(self.volumes, "longitudinal_volumes", ylabel="Volume")

    @task
    def log_longitudinal(self):
        self._plot_longitudinal(
            self.log_volumes, "log_longitudinal_volumes", ylabel="Log volume"
        )

    @task
    def trajectory_summary(self):
        self._plot_trajectory_summary(self.volumes, "trajectory_summary")

    @task
    def log_trajectory_summary(self):
        self._plot_trajectory_summary(self.log_volumes, "log_trajectory_summary")

    @task
    def trajectory_summary_lr(self):
        self._plot_trajectory_summary_lr(self.volumes, "trajectory_summary_lr")

    @task
    def log_trajectory_summary_lr(self):
        self._plot_trajectory_summary_lr(self.log_volumes, "log_trajectory_summary_lr")

    def _save_figure(self, fig, name):
        fig.savefig(self.figures_dir / f"{name}.pdf")
        plt.close(fig)

    def _plot_longitudinal(self, volumes, name, ylabel):
        fig, _ = plot_grid(
            volumes,
            plot_layers,
            plot_fnc=plot_volumes_over_time,
            layer_kwargs={
                "case": {
                    "kind": "plot",
                    "marker": "o",
                    "markersize": 3,
                    "alpha": 0.5,
                    "colors": self.colors,
                    "legend": True,
                    "ylabel": ylabel,
                },
                "control": {
                    "kind": "scatter",
                    "marker": "x",
                    "alpha": 0.25,
                    "colors": self.colors,
                    "ylabel": ylabel,
                },
            },
            legend=False,
            figsize=(10, 16),
        )

        self._save_figure(fig, name)

    def _plot_trajectory_summary(self, volumes, name):
        def _select(_, grouped):
            return (
                grouped.map_values(
                    lambda dataset: dataset.split_outer().map_values(fit_trajectory)
                ),
            )

        fig, _ = plot_grid(
            volumes,
            plot_layers,
            select=_select,
            plot_fnc=plot_trajectory_summary,
            layer_kwargs={
                "case": {"colors": self.colors, "legend": True},
                "control": {"colors": self.colors, "alpha": 0.25},
            },
            legend=False,
            sharex=False,
            figsize=(10, 16),
        )

        self._save_figure(fig, name)

    def _plot_trajectory_summary_lr(self, volumes, name):
        summaries = {
            struct: grouped.map_values(
                lambda dataset: dataset.split_outer().map_values(fit_trajectory)
            )
            for struct, grouped in volumes.items()
        }

        paired = pair_hemispheres(summaries)

        layers = {
            struct: {
                f"{group}_{side}": data
                for side, hemisphere in hemispheres.items()
                for group, data in hemisphere.items()
            }
            for struct, hemispheres in paired.items()
        }

        layer_kwargs = {
            "case_left": {
                "colors": self.colors,
                "marker": "<",
                "legend": True,
            },
            "case_right": {
                "colors": self.colors,
                "marker": ">",
            },
            "control_left": {
                "colors": self.colors,
                "marker": "<",
            },
            "control_right": {
                "colors": self.colors,
                "marker": ">",
            },
        }

        fig, _ = plot_grid(
            layers,
            plot_layers,
            plot_fnc=plot_trajectory_summary,
            layer_kwargs=layer_kwargs,
            legend=False,
            figsize=(8, 10),
            sharex=False,
            legend_wrap=1,
        )

        layer_handles = [
            make_legend_handle(name_, kwargs_)
            for name_, kwargs_ in {
                "left": {"marker": "<"},
                "right": {"marker": ">"},
            }.items()
        ]

        fig.legend(
            handles=layer_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(layer_handles),
            frameon=False,
        )

        self._save_figure(fig, name)
