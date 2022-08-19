#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:00:09 2022

@author: joan
https://plotly.com/python/visualizing-mri-volume-slices/
He instalat plotly fent 
conda install -c plotly plotly=5.10.0
"""

import numpy as np

# Import data
fn = 'snapshots/dataset_snapshot_1.npz'
f = np.load(fn)
volume = f['x']
#label = f['y']
feature = 4 # TKE_normalized : Turbulent Kinetic Energy
title = 'Turbulent Kinetic Energy'
volume = volume[:, :, :, feature] 
volume = np.swapaxes(volume, 2, 0)
r, c = volume[0].shape
nb_frames = len(volume)

# Define frames
import plotly.graph_objects as go

fig = go.Figure(frames=[go.Frame(data=go.Surface(
    z=((nb_frames-1)/10. - k * 0.1) * np.ones((r, c)),
    surfacecolor=np.flipud(volume[nb_frames - k - 1]),
    cmin=0, cmax=volume.max()
    ),
    name=str(k) # you need to name the frame for the animation to behave properly
    )
    for k in range(nb_frames)])

# Add data to be displayed before animation starts
fig.add_trace(go.Surface(
    z=(nb_frames-1)/10. * np.ones((r, c)),
    surfacecolor=np.flipud(volume[nb_frames - 1]),
    colorscale='Gray',
    cmin=0, cmax=volume.max(),
    colorbar=dict(thickness=20, ticklen=4)
    ))


def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [[f.name], frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(fig.frames)
                ],
            }
        ]

# Layout
fig.update_layout(
         title=title,
         width=600,
         height=600,
         scene=dict(
                    zaxis=dict(range=[-0.1, nb_frames/10], autorange=False),
                    aspectratio=dict(x=1, y=1, z=1),
                    ),
         updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, frame_args(50)],
                        "label": "&#9654;", # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;", # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
         ],
         sliders=sliders
)

#fig.show(block=False)
fig.write_html('vis-snapshot.html', auto_open=True)