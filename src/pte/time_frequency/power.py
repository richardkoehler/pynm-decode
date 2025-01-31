"""Functions for calculating and handling band-power."""

from pathlib import Path
from typing import Literal

import matplotlib.figure
import mne
import mne_bids
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt

import pte


def plot_power(
    power: mne.time_frequency.AverageTFR,
    title: str | None = None,
    fname: str | Path | None = None,
    show: bool = True,
    **kwargs_plot,
) -> matplotlib.figure.Figure:
    """Plot single MNE AverageTFR object."""
    fig, axes = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(4.75, 3.75),
        tight_layout=True,
    )
    fig = power.plot(
        picks="all",
        cmap="viridis",
        title=title,
        axes=axes,
        show=show,
        verbose=False,
        **kwargs_plot,
    )[0]
    if show:
        plt.show(block=True)
    if fname:
        fig.savefig(
            fname=fname,
            dpi=300,
            bbox_inches="tight",
        )
    return fig


def smooth_power(
    power: mne.time_frequency.AverageTFR,
    smoothing_type: Literal["gaussian", "median"] = "gaussian",
    **kwargs,
) -> mne.time_frequency.AverageTFR:
    """Smooth data in AverageTFR object using scipy smoothing filters."""
    power = power.copy()
    for i, data in enumerate(power.data):
        power.data[i] = smooth_2d_array(
            data=data,
            smoothing_type=smoothing_type,
            **kwargs,
        )
    return power


def smooth_2d_array(
    data: np.ndarray,
    smoothing_type: Literal["gaussian", "median"] = "gaussian",
    **kwargs,
) -> np.ndarray:
    """Smooth 2D data using scipy smoothing filters."""
    if smoothing_type == "gaussian":
        if "sigma" not in kwargs:
            kwargs["sigma"] = 1
        data_out = scipy.ndimage.gaussian_filter(
            input=data, mode="reflect", **kwargs
        )
    elif smoothing_type == "median":
        if "size" not in kwargs:
            kwargs["size"] = 1
        data_out = scipy.ndimage.median_filter(
            input=data,
            mode="reflect",
            **kwargs,
        )
    else:
        raise ValueError(
            "Not a valid ``smoothing_type``. Got:" f" {smoothing_type}."
        )
    return data_out


def apply_baseline(
    power: mne.time_frequency.AverageTFR,
    baseline: (
        tuple[int | float | None, int | float | None] | np.ndarray | None
    ) = (None, None),
    mode: str = "percent",
) -> mne.time_frequency.AverageTFR:
    """Apply baseline correction given interval or custom values."""
    if baseline is None:
        return power
    if not isinstance(baseline, np.ndarray):
        return power.apply_baseline(
            baseline=baseline, mode=mode, verbose=False
        )
    return apply_baseline_array(power=power, baseline=baseline, mode=mode)


# This code is adapted from MNE-Python version 0.24.1 (mne.baseline.rescale())
# Copyright © 2011-2022, authors of MNE-Python
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the copyright holder nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.


# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
def apply_baseline_array(
    power: mne.time_frequency.AverageTFR | np.ndarray,
    baseline: np.ndarray,
    mode: str = "percent",
) -> mne.time_frequency.AverageTFR:
    """Apply baseline correction with array of baseline values."""
    if isinstance(power, np.ndarray):
        data = power.copy()
    elif isinstance(power, mne.time_frequency.AverageTFR):
        data = power.data.copy()
    else:
        raise ValueError(
            "Unknown type for `power`. Must be a numpy array or"
            f" mne.time_frequency.AverageTFR. Got: {type(power)}."
        )
    if not baseline.ndim == data.ndim:
        raise ValueError(
            "If `baseline` is an array of values, it must have"
            " the same number of dimensions as `power`. Number of dimensions"
            f" `baseline`: {baseline.ndim}. Number of dimensions `power`:"
            f" {data.ndim}."
        )
    if not baseline.shape[:-1] == data.shape[:-1]:
        raise ValueError(
            "If `baseline` is an array of values, all dimensions but the last"
            " must be the same as `power`. Shape of `baseline`:"
            f" {baseline.shape}. Shape of `power`: {data.shape}."
        )
    mean = np.mean(baseline, axis=-1, keepdims=True)

    if mode == "mean":
        data -= mean

    elif mode == "ratio":
        data /= mean

    elif mode == "logratio":
        data /= mean
        np.log10(data, out=data)

    elif mode == "percent":
        data -= mean
        data /= mean

    elif mode == "zscore":
        data -= mean
        data /= np.std(baseline, axis=-1, keepdims=True)

    elif mode == "zlogratio":
        data /= mean
        np.log10(data, out=data)
        data /= np.std(baseline, axis=-1, keepdims=True)

    else:
        raise ValueError(f"Unknown baseline correction mode: {mode}.")

    power.data = data

    return power


# This code is adapted from MNE-Python version 0.24.1 (mne.baseline.rescale())
# For license information see above.
def _get_baseline_indices(
    baseline: tuple[int | float | None, int | float | None],
    times: np.ndarray,
) -> tuple[int, int]:
    """Get baseline indices from times array."""
    bmin, bmax = baseline
    if bmin is None:
        imin = 0
    else:
        baseline_indices = np.where(times >= bmin)[0]
        if len(baseline_indices) == 0:
            raise ValueError(
                f"The baseline values are not within the limits of the time"
                f" range. Baseline values: {baseline}. Time range: {times[0]}"
                f" to {times[-1]}."
            )
        imin = int(baseline_indices[0])
    if bmax is None:
        imax = len(times)
    else:
        baseline_indices = np.where(times <= bmax)[0]
        if len(baseline_indices) == 0:
            raise ValueError(
                f"The baseline values are not within the limits of the time"
                f" range. Baseline values: {baseline}. Time range: {times[0]}"
                f" to {times[-1]}."
            )
        imax = int(baseline_indices[-1]) + 1
    if imin >= imax:
        raise ValueError(
            f"Bad rescaling slice ({imin}:{imax}) from time values"
            f" {bmin}, {bmax}."
        )
    return imin, imax


def get_baseline(
    powers: list[mne.time_frequency.AverageTFR],
    picks: str | list[str] | slice,
    baseline: tuple[int | float | None, int | float | None] = (None, None),
) -> list[np.ndarray]:
    """Get baseline array of given list of AverageTFRs."""
    if not isinstance(powers, list):
        powers = [powers]

    baselines = []
    for power in powers:
        baseline_array = power.copy().pick(picks=picks)
        imin, imax = _get_baseline_indices(
            baseline=baseline, times=power.times
        )
        baseline_array = power.data[..., imin:imax]
        baselines.append(baseline_array)

    return baselines


def average_power(
    powers: (
        list[mne.time_frequency.AverageTFR] | mne.time_frequency.AverageTFR
    ),
    picks: str | list[str] | slice,
    baseline: (
        tuple[int | float | None, int | float | None]
        | list[np.ndarray]
        | np.ndarray
        | None
    ) = (None, None),
    baseline_mode: str | None = "zscore",
    clip: int | float | None = None,
) -> mne.time_frequency.AverageTFR:
    """Return power averaged over given channel types or picks."""
    if not isinstance(powers, list):
        powers = [powers]
    if isinstance(baseline, np.ndarray):
        baseline = [baseline]
    if isinstance(baseline, list) and not len(baseline) == len(powers):
        raise ValueError(
            "If numpy arrays are provided for"
            "custom baseline correction, the same number of numpy arrays"
            " and `power` (TFR) objects must be provided. Got:"
            f" {len(baseline)} baseline arrays and {len(powers)} powers."
        )

    power_all = None
    power_all_files = []
    for i, power in enumerate(powers):
        power = power.copy().pick(picks=picks)
        if baseline is not None:
            if baseline_mode is None:
                raise ValueError(
                    "If baseline correction is performed, `baseline_mode`"
                    "must not be `None`."
                )
            if isinstance(baseline, list):
                baseline = baseline[i]
            power = apply_baseline(
                power=power, baseline=baseline, mode=baseline_mode
            )  # type: ignore
        df_power = power.to_data_frame(picks=picks)
        freqs = power.freqs  # type: ignore
        power_all_freqs = []
        for freq in freqs:
            power_single_freq = (
                df_power[df_power["freq"] == freq]
                .sort_values(by="time", axis="index")
                .drop(columns=["freq", "time"])
                .to_numpy()
            )
            # reject artifacts by clipping
            if clip is not None:
                power_single_freq = power_single_freq.clip(
                    min=clip * -1, max=clip
                )
            power_all_freqs.append(power_single_freq)
        # Average across all channels
        power_all_files.append(np.stack(power_all_freqs, axis=0).mean(axis=-1))
    power_array_all = np.expand_dims(
        np.stack(power_all_files, axis=0).mean(axis=0), axis=0
    )
    if clip is not None:
        power_array_all = power_array_all.clip(min=clip * -1, max=clip)
    power = powers[0]
    info = mne.create_info(
        ch_names=1, sfreq=power.info["sfreq"], ch_types="misc", verbose=False
    )
    power_all = mne.time_frequency.AverageTFR(
        info=info,
        data=power_array_all,
        times=power.times,
        freqs=power.freqs,
        nave=len(powers),
        comment=power.comment,
        method=power.method,
    )
    return power_all


def load_power(
    files: list[Path | str], verbose: bool | str | int | None = False
) -> list[mne.time_frequency.AverageTFR] | list[mne.time_frequency.EpochsTFR]:
    """Load power from *-tfr.h5 files."""
    powers = []
    for file in files:
        power = mne.time_frequency.read_tfrs(file)
        if isinstance(power, list):
            if verbose:
                print("Only returning first object from tfr file.")
            power = power[0]
        powers.append(power)
    return powers


def morlet_from_epochs(
    epochs: mne.Epochs,
    n_cycles: int = 7,
    freqs: str | np.ndarray = "auto",
    average: bool = True,
    n_jobs: int = -1,
    picks: str = "all",
    decim_power: str | int | float = "auto",
    verbose: str | bool = "ERROR",
    **kwargs,
) -> mne.time_frequency.AverageTFR | mne.time_frequency.EpochsTFR:
    """Calculate power with MNE's Morlet transform and sensible defaults."""
    if isinstance(freqs, str):
        if freqs != "auto":
            raise ValueError(
                f"If freqs is a string, it must be `auto`. Got: {freqs}"
            )
        upper_freq = min(epochs.info["sfreq"] / 2, 200.0)
        freqs = np.arange(1.0, upper_freq)

    if "decim" not in kwargs:
        if isinstance(decim_power, str):
            if decim_power != "auto":
                raise ValueError(
                    "If decim_power is a string, it must be `auto`. Got:"
                    f" {decim_power}"
                )
            decim = int(epochs.info["sfreq"] / 100)
        else:
            decim = int(decim_power)
        kwargs["decim"] = decim

    power = epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        picks=picks,
        average=average,
        verbose=verbose,
        n_jobs=n_jobs,
        n_cycles=n_cycles,
        **kwargs,
    )
    return power


def epochs_from_raw(
    raw: mne.io.BaseRaw,
    events_trial_onset: str | list[str] | list[tuple[str, str]],
    events_trial_end: str | list[str] | list[tuple[str, str]] | None = None,
    tmin: int | float = -6,
    tmax: int | float = 6,
    baseline: tuple | None = None,
    min_distance_trials: int | float = 0,
    verbose: bool | str | int | None = True,
    **kwargs,
) -> mne.Epochs:
    """Return epochs from given events."""
    events, _ = get_events(raw=raw, event_picks=events_trial_onset)
    epochs = mne.Epochs(
        raw=raw,
        events=events,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        verbose=verbose,
        **kwargs,
    )
    if min_distance_trials:
        events_end = None
        if events_trial_end:
            events_end, _ = get_events(raw, event_picks=events_trial_end)
        epochs = discard_epochs(
            epochs=epochs,
            events_begin=events,
            events_end=events_end,
            min_distance_events=min_distance_trials,
        )
    return epochs


def get_events(
    raw: mne.io.BaseRaw,
    event_picks: str | list[str] | list[tuple[str, str]],
) -> tuple[np.ndarray, dict]:
    """Get events from given Raw instance and event id."""
    if isinstance(event_picks, str):
        event_picks = [event_picks]
    events = None
    for event_pick in event_picks:
        if isinstance(event_pick, str):
            event_id = {event_pick: 1}
        else:
            event_id = {event_pick[0]: 1, event_pick[1]: -1}
        try:
            events, _ = mne.events_from_annotations(
                raw=raw,
                event_id=event_id,
                verbose=True,
            )
            return events, event_id
        except ValueError as error:
            print(error)
    _, event_id_found = mne.events_from_annotations(
        raw=raw,
        verbose=False,
    )
    raise ValueError(
        f"None of the given `event_picks´ found: {event_picks}."
        f"Possible events: {*event_id_found.keys(),}"
    )


def discard_epochs(
    epochs: mne.Epochs,
    events_begin: np.ndarray,
    min_distance_events: int | float,
    events_end: np.ndarray | None = None,
    inplace: bool = True,
) -> mne.Epochs:
    """Discard epochs based on minimal distance between event onset and end."""
    if events_end is not None:
        events = np.sort(np.hstack((events_begin[:, 0], events_end[:, 0])))
        event_diffs = np.diff(events)[1::2]
    else:
        events = events_begin[:, 0]
        event_diffs = np.diff(events)
    dist_small = np.where(
        event_diffs <= min_distance_events * epochs.info["sfreq"]
    )[0]
    drop_indices = np.unique(np.concatenate((dist_small, dist_small + 1)))
    if not inplace:
        epochs = epochs.copy()
    return epochs.drop(indices=drop_indices, reason="min_distance_events")


def power_from_bids(
    bids_path: mne_bids.BIDSPath,
    nm_channels_dir: Path,
    events_trial_onset: str | list[str] | list[tuple[str, str]],
    events_trial_end: str | list[str] | list[tuple[str, str]] | None = None,
    min_distance_trials: int | float = 0,
    bad_epochs_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    kwargs_preprocess: dict | None = None,
    kwargs_epochs: dict | None = None,
    kwargs_power: dict | None = None,
    verbose: bool | str | int | None = False,
) -> mne.time_frequency.AverageTFR | mne.time_frequency.EpochsTFR | None:
    """Calculate power from single file."""
    print(f"File: {bids_path.basename}")
    raw = mne_bids.read_raw_bids(bids_path, verbose=verbose)
    try:
        if kwargs_preprocess is None:
            kwargs_preprocess = {}
        raw = pte.preprocessing.preprocess(
            raw=raw,
            nm_channels_dir=nm_channels_dir,
            filename=bids_path,
            pick_used_channels=True,
            verbose=verbose,
            **kwargs_preprocess,
        )
    except ValueError as error:
        if "No valid channels found" not in str(error):
            raise
        print(error)
        return None

    if kwargs_epochs is None:
        kwargs_epochs = {}
    epochs = epochs_from_raw(
        raw=raw,
        events_trial_onset=events_trial_onset,
        events_trial_end=events_trial_end,
        min_distance_trials=min_distance_trials,
        **kwargs_epochs,
    )

    if bad_epochs_dir is not None:
        try:
            bad_epochs_df = pte.filetools.get_bad_epochs(
                filename=bids_path,
                bad_epochs_dir=bad_epochs_dir,
            )
            if bad_epochs_df is not None:
                bad_epochs = bad_epochs_df.event_id.to_numpy()
                bad_indices = np.array(
                    [
                        idx
                        for idx, event in enumerate(epochs.selection)
                        if event in bad_epochs
                    ]
                )
                epochs = epochs.drop(indices=bad_indices)
        except FileNotFoundError as e:
            print(e, "\nNot gathering bad epochs for the current file.")

    if kwargs_power is None:
        kwargs_power = {}
    if "freqs" not in kwargs_power:
        freqs = np.arange(1, 200, 1)
    else:
        freqs = kwargs_power.pop("freqs")
    power = morlet_from_epochs(
        epochs=epochs,
        freqs=freqs,
        **kwargs_power,
    )

    if out_dir:
        fname = Path(out_dir) / (str(bids_path.fpath.stem) + "_tfr.h5")
        power.save(fname=fname, verbose=verbose, overwrite=True)
    return power
