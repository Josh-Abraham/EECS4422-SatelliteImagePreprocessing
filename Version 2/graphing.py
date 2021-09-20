import gc
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import plotly.graph_objects as go



def getGraphOne():
    colors = ['#34ebba', '#f5bf42', '#a8326f']
    x = ['Canny', 'HED Model', 'SPEED <br> & Canny']
    ture_pos = 0.6189645, 0.6530904, 0.7434262
    false_pos = 0.647972, 0.4431435, 0.3385871 

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=ture_pos, marker_color=colors))

    fig.update_layout(yaxis_title='True Positive Score', title_x=0.5, font=dict(
            family="Times, monospace",
            size=20
        ))

    fig.update_xaxes(
            title_text = "Edge Detector",
            tickfont=dict(size=22),
            title_standoff = 25)

    fig.update_yaxes(
            tickfont=dict(size=22),
            title_standoff = 25)
    fig.show()


def getAblation():
    colors = ['#a8326f', 'lightslategray',  'lightslategray',  'lightslategray',  'lightslategray']
    x = ['Full Pipeline', 'No Conditional <br> Blur', 'No Conditional <br>Contrast', 'No FHH', 'No Anisotropic <br> Diffusion']
    true_pos = 0.7434262, 0.727965, 0.7347713, 0.6277815, 0.690303
    false_pos = 0.3385871, 0.346641320, 0.3666355, 0.5194913, 0.546414

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=false_pos, marker_color=colors))

    fig.update_layout(yaxis_title='False Positive Score', title_x=0.5, font=dict(
            family="Times, monospace",
            size=20
        ))

    fig.update_xaxes(
            title_text = "Augmented Pipeline",
            tickfont=dict(size=20),
            title_standoff = 25)

    fig.update_yaxes(
            tickfont=dict(size=22),
            title_standoff = 25)
    fig.show()


getGraphOne()