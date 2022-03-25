import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import torch
import warnings


def plot_traj(gaussian, epoch=0, n_states=4):
    n_states = min(n_states, gaussian.event_shape[0])
    fig = go.Figure()
    xrange = np.arange(gaussian.batch_shape[-1])
    for i in range(n_states):
        fig.add_scatter(x=np.concatenate([xrange, xrange[::-1]], axis=0),
                        y=np.concatenate([gaussian.mean.detach().numpy()[epoch, :, i] + gaussian.stddev.detach().numpy()[epoch, :, i],
                                          gaussian.mean.detach().numpy()[epoch, ::-1, i] - gaussian.stddev.detach().numpy()[epoch, ::-1, i]], axis=0),
                        mode='lines', fill='toself', legendgroup=f'x[{i}]', showlegend=False,
                        line_color=px.colors.qualitative.T10[i], fillcolor=px.colors.qualitative.T10[i], opacity=0.15)
    for i in range(n_states):
        fig.add_scatter(y=gaussian.mean.detach().numpy()[epoch, :, i], mode='lines', name=f'x[{i}]', legendgroup=f'x[{i}]',
                        line_color=px.colors.qualitative.T10[i])
    return fig


def handle_nan(x: torch.Tensor, where: str = '', what: str = 'A variable'):
    if torch.isnan(x).any():
        warnings.warn(f'{where}: {what} contains NaN values!'
                      '\n Applied corrections may lead to numerical instabilities!', RuntimeWarning)
        return torch.nan_to_num_(x)
    else:
        return x
