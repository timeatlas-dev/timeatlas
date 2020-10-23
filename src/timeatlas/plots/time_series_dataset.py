from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
import plotly.io as pio

if TYPE_CHECKING:
    from timeatlas.time_series_dataset import TimeSeriesDataset

pio.templates.default = "plotly_white"


def line_plot(tsd: 'TimeSeriesDataset', *args, **kwargs) -> Any:
    """Plot a TimeSeriesDataset with Plotly

    Args:
        tsd: TimeSeriesDataset

    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()
    for ts in tsd:
        fig.add_trace(
            go.Scatter(x=ts.index,
                       y=ts.values,
                       mode='lines'))
    return fig
