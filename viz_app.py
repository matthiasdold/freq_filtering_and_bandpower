import dash
import dash_bootstrap_components as dbc
import mne
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from dash import Input, Output, dcc, html
from scipy.signal import butter, hilbert, sosfilt, sosfiltfilt, welch

mne.set_log_level("WARNING")

pio.templates.default = "plotly_white"

# get examplate data from mne --> the code to generate
#
# from mne_lsl.datasets import sample
# raw = (
#     mne.io.read_raw(sample.data_path() / "sample-ant-raw.fif", preload=True)
#     .resample(256)
#     .pick_channels(["Fp1"])
# )  # resample for speed
#
# raw.save("./assets/sample-ant-Fp1-raw.fif")
raw = mne.io.read_raw("./assets/sample-ant-Fp1-raw.fif", preload=True)

x = raw.get_data()[0, :].T * 1e6
x -= x.mean()
ch_name = raw.info["ch_names"][0]
sfreq = raw.info["sfreq"]
t = np.arange(0, len(x) / sfreq, 1 / sfreq)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------
controls_freq = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("Freq. range [Hz]"),
                dcc.RangeSlider(
                    id="freq-range-filtering",
                    min=1,
                    max=100,
                    step=1,
                    value=[1, 100],
                    marks={i: str(i) for i in range(1, 101, 20)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Time range [s]"),
                dcc.RangeSlider(
                    id="time-range-filtering",
                    min=0,
                    max=t.max(),
                    step=0.2,
                    value=[0, t.max()],
                    marks={i: str(i) for i in range(0, int(t.max() + 1), 20)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Filter order"),
                dbc.Input(
                    id="filter-order-filtering",
                    type="number",
                    min=1,
                    max=10,
                    step=1,
                    value=2,
                ),
            ]
        ),
        html.Hr(),
        html.Div(
            [
                dbc.Label("Components"),
                dbc.Checklist(
                    id="components-filtering",
                    options=["raw", "acausal", "causal"],
                    value=["raw", "acausal", "causal"],
                    inline=True,
                ),
            ]
        ),
    ],
    body=True,
)


controls_bp = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("Freq. range [Hz]"),
                dcc.RangeSlider(
                    id="freq-range-bp",
                    min=1,
                    max=100,
                    step=1,
                    value=[12, 15],
                    marks={i: str(i) for i in range(1, 101, 20)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Time range [s]"),
                dcc.RangeSlider(
                    id="time-range-bp",
                    min=0,
                    max=10,  # short time window for BP, as compute becomes unfeasible otherwise
                    step=0.2,
                    value=[0, 5],
                    marks={i: str(i) for i in range(0, int(t.max() + 1), 1)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Rolling window lenght [s]"),
                dcc.Slider(
                    id="rolling-range-bp",
                    min=0,
                    max=10,
                    step=0.2,
                    value=4,
                    marks={i: str(i) for i in range(0, 11, 1)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Filter order"),
                dbc.Input(
                    id="filter-order-bp",
                    type="number",
                    min=1,
                    max=10,
                    step=1,
                    value=2,
                ),
            ]
        ),
        html.Hr(),
        html.Div(
            [
                dbc.Label("Components"),
                dbc.Checklist(
                    id="components-bp",
                    options=["welch", "multitaper", "hilbert", "raw"],
                    value=["welch", "multitaper", "hilbert"],
                    inline=True,
                ),
            ]
        ),
    ],
    body=True,
)


app.layout = dbc.Container(
    [
        dbc.Tabs(
            [
                dbc.Tab(
                    label="Frequency filtering",
                    tab_id="freq_filt",
                    children=[
                        dbc.Spinner(
                            [
                                dcc.Markdown(
                                    f"Frequency filtering of {ch_name}, raw {sfreq=}Hz"
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(controls_freq, md=4),
                                        dbc.Col(dcc.Graph(id="freq-graph"), md=8),
                                    ],
                                    align="center",
                                ),
                            ],
                            delay_show=100,
                        ),
                    ],
                ),
                dbc.Tab(
                    label="Bandpower estimation",
                    tab_id="bp_estimation",
                    children=[
                        dbc.Spinner(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(controls_bp, md=4),
                                        dbc.Col(dcc.Graph(id="bp-graph"), md=8),
                                    ],
                                    align="center",
                                ),
                            ],
                            delay_show=100,
                        ),
                    ],
                ),
            ],
            id="tabs",
            active_tab="freq_filt",
        ),
        # we wrap the store and tab content with a spinner so that when the
        # data is being regenerated the spinner shows. delay_show means we
        # don't see the spinner flicker when switching tabs
    ]
)

# ----------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------


@app.callback(
    Output("freq-graph", "figure"),
    Input("freq-range-filtering", "value"),
    Input("time-range-filtering", "value"),
    Input("filter-order-filtering", "value"),
    Input("components-filtering", "value"),
)
def generate_freq_figure(freq_range, time_range, filter_order, components):
    global x, t, sfreq

    time_mask = (t >= time_range[0]) & (t <= time_range[1])
    xp = x[time_mask]
    tp = t[time_mask]

    # create n-th order
    sos = butter(filter_order, freq_range, btype="bandpass", output="sos", fs=sfreq)

    xp_acausal = sosfiltfilt(sos, xp)
    xp_causal = sosfilt(sos, xp)

    fig = go.Figure()
    for comp in components:
        match comp:
            case "raw":
                fig.add_trace(go.Scatter(x=tp, y=xp, name="raw"))
            case "acausal":
                fig.add_trace(go.Scatter(x=tp, y=xp_acausal, name="acausal"))
            case "causal":
                fig.add_trace(
                    go.Scatter(x=tp, y=xp_causal, mode="lines", name="causal")
                )

    fig = fig.update_traces(
        # mode="lines+markers",
        line=dict(width=1),
    )
    fig = fig.update_layout(
        title=f"Signal between [{tp[0]:.2f}-{tp[-1]:.2f}]s, {freq_range}Hz",
        xaxis_title="Time [s]",
        yaxis_title=f"{ch_name} [μV]",
        legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="center", x=0.5),
    )

    return fig


@app.callback(
    Output("bp-graph", "figure"),
    Input("freq-range-bp", "value"),
    Input("time-range-bp", "value"),
    Input("filter-order-bp", "value"),
    Input("rolling-range-bp", "value"),
    Input("components-bp", "value"),
)
def generate_bp_figure(
    freq_range, time_range, filter_order, rolling_range_bp, components
):
    global x, t, sfreq

    time_mask = (t >= time_range[0]) & (t <= time_range[1])
    xp = x[time_mask]
    tp = t[time_mask]

    # bandpower via welch and mne.time_frequency.psd_array_multitaper
    rolling_range_bp_idx = int(rolling_range_bp * sfreq)
    x_welch = np.zeros(xp.shape)
    x_multitaper = np.zeros(xp.shape)

    sos = butter(filter_order, freq_range, btype="bandpass", output="sos", fs=sfreq)
    zi = np.zeros((sos.shape[0], 2))  # type: ignore
    x_filtered = np.zeros(xp.shape)
    x_hilbert = np.zeros(xp.shape)

    # warm up filter up to the first point that the rolling window would be able to produce an output
    xf, zi = sosfilt(sos, xp[:rolling_range_bp_idx], zi=zi)
    #
    # using a loop to simulate procedural processing
    print(f"Processing for {(len(xp) - rolling_range_bp_idx)=} steps")
    for idx in range(rolling_range_bp_idx, len(xp)):
        x_window = xp[idx - rolling_range_bp_idx : idx]

        # calculate, windowed approach
        # -- Welch
        fw, Pxx_welch = welch(x_window, sfreq, nperseg=sfreq)
        fidx = (fw >= freq_range[0]) & (fw <= freq_range[1])
        x_welch[idx] = np.mean(Pxx_welch[fidx])

        # -- Multitaper
        psd, _ = mne.time_frequency.psd_array_multitaper(  # type: ignore
            x_window,
            sfreq,
            fmin=freq_range[0],
            fmax=freq_range[1],
            normalization="full",
        )
        x_multitaper[idx] = np.mean(psd)
        #
        # -- hilbert
        x_filtered[idx], zi = sosfilt(
            sos,
            xp[idx - 1 : idx],
            zi=zi,
        )
        x_hilbert[idx] = np.abs(hilbert(x_filtered[idx - rolling_range_bp_idx : idx]))[-1]  # type: ignore

    fig = go.Figure()
    slice_plot = slice(rolling_range_bp_idx - 100, len(xp))
    for comp in components:
        match comp:
            case "welch":
                fig.add_trace(
                    go.Scatter(x=tp[slice_plot], y=x_welch[slice_plot], name="welch")
                )
            case "multitaper":
                fig.add_trace(
                    go.Scatter(
                        x=tp[slice_plot], y=x_multitaper[slice_plot], name="multitaper"
                    )
                )
            case "hilbert":
                fig.add_trace(
                    go.Scatter(
                        x=tp[slice_plot],
                        y=x_hilbert[slice_plot],
                        mode="lines",
                        name="hilbert",
                    )
                )
            case "raw":
                # filter and crop the raw just for visualization
                xp_filtered = (
                    raw.copy().pick_channels([ch_name]).filter(*freq_range).get_data()
                )[0][time_mask] * 1e6

                fig.add_trace(
                    go.Scatter(
                        x=tp[slice_plot],
                        y=xp_filtered[slice_plot],
                        mode="lines",
                        name="raw_filtered",
                    )
                )

    fig = fig.update_traces(
        # mode="lines+markers",
        line=dict(width=1),
    )
    fig = fig.update_layout(
        title=f"Bandpower between [{tp[0]:.2f}-{tp[-1]:.2f}]s, {freq_range}Hz",
        xaxis_title="Time [s]",
        yaxis_title=f"{ch_name} bandpower [μV**2/Hz]",
    )

    return fig


if __name__ == "__main__":
    app.title = "EEG filtering"
    app.run(debug=True, port=8888)
