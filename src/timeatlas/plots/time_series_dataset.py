from typing import TYPE_CHECKING, Any
import plotly.graph_objects as go

if TYPE_CHECKING:
    from timeatlas.time_series_dataset import TimeSeriesDataset


def line(tsd: 'TimeSeriesDataset') -> Any:
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
