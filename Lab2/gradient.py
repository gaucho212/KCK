import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import colorsys


def hsv2rgb(h, s, v):
    
    (r,g,b) = colorsys.hsv_to_rgb(h,s,v)

    return (r,g,b)


def gradient_rgb_bw(v):
    return (v, v, v)


def gradient_rgb_gbr(v):
    if v < 0.5:
        t = v / 0.5
        return (0.0, 1.0 - t, t)
    else:
        t = (v - 0.5) / 0.5
        return (t, 0.0, 1.0 - t)


def gradient_rgb_gbr_full(v):
    if v < 0.25:
        t = v / 0.25
        return (0.0, 1.0, t)           
    elif v < 0.5:
        t = (v - 0.25) / 0.25
        return (0.0, 1.0 - t, 1.0)     
    elif v < 0.75:
        t = (v - 0.5) / 0.25
        return (t, 0.0, 1.0)           
    else:
        t = (v - 0.75) / 0.25
        return (1.0, 0.0, 1.0 - t)     


def gradient_rgb_wb_custom(v):
    points = [
        (1.0, 1.0, 1.0),  
        (1.0, 0.0, 1.0),  
        (0.0, 0.0, 1.0),  
        (0.0, 1.0, 1.0),  
        (0.0, 1.0, 0.0),  
        (1.0, 1.0, 0.0),  
        (1.0, 0.0, 0.0),  
        (0.0, 0.0, 0.0)   
    ]
    num_segments = len(points) - 1
    v = max(0.0, min(1.0, v))
    val = v * num_segments
    idx = min(int(val), num_segments - 1)
    t = val - idx
    p1 = points[idx]
    p2 = points[idx + 1]

    return (
        p1[0] + t * (p2[0] - p1[0]),
        p1[1] + t * (p2[1] - p1[1]),
        p1[2] + t * (p2[2] - p1[2])
    )


def gradient_hsv_bw(v):
    return hsv2rgb(0.0, 0.0, v)


def gradient_hsv_gbr(v):
    h = 1.0 / 3.0 + v * (2.0 / 3.0)
    return hsv2rgb(h, 1.0, 1.0)


def gradient_hsv_unknown(v):
    h = (1.0 - v) / 3.0   
    s = 0.5               
    val = 1.0
    return hsv2rgb(h, s, val)


def gradient_hsv_custom(v):
    h = 1-(0.65 * v)          
    s = 1.0 - (v ** 1.5)   
    val = 1.0
    return hsv2rgb(h, s, val)


def plot_color_gradients(gradients, names):
    rc('legend', fontsize=10)

    column_width_pt = 400
    pt_per_inch = 72
    size = column_width_pt / pt_per_inch

    fig, axes = plt.subplots(nrows=len(gradients), sharex=True, figsize=(size, 0.75 * size))
    fig.subplots_adjust(top=0.98, bottom=0.08, left=0.28, right=0.95)

    for ax, gradient, name in zip(axes, gradients, names):
        img = np.zeros((2, 1024, 3))
        for i, v in enumerate(np.linspace(0, 1, 1024)):
            img[:, i] = gradient(v)

        im = ax.imshow(img, aspect='auto')
        im.set_extent([0, 1, 0, 1])
        ax.yaxis.set_visible(False)

        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.26
        y_text = pos[1] + pos[3] / 2.0
        fig.text(x_text, y_text, name, va='center', ha='left', fontsize=9)

    fig.savefig('my-gradients.pdf')
    plt.show()


def toname(g):
    return g.__name__.replace('gradient_', '').replace('_', '-').upper()


if __name__ == '__main__':
    gradients = (
        gradient_rgb_bw,
        gradient_rgb_gbr,
        gradient_rgb_gbr_full,
        gradient_rgb_wb_custom,
        gradient_hsv_bw,
        gradient_hsv_gbr,
        gradient_hsv_unknown,
        gradient_hsv_custom
    )

    plot_color_gradients(gradients, [toname(g) for g in gradients])