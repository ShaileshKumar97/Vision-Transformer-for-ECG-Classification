import argparse

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wfdb

from plot import lead_index, plot_ax
from vit import VisionTransformer


def main(file_path: str, wights_path: str, lead: int):
    """
    This function plots the attention scores on the ECG signal to interpret how our model focus on (or giving attention) to
    ecg signal.

    Args:
        file_path (str): file path to a single wfdb file.
        weights_path (str): Trained model's weights path.
        lead (int): Lead index between 1 and 12.
    """

    vit = VisionTransformer(patch_size=20, hidden_size=768, depth=6, num_heads=6, mlp_dim=256, num_classes=2)
    vit.load_weights(wights_path)

    record = wfdb.rdrecord(file_path)

    attn = vit.get_last_selfattention(tf.expand_dims(record.p_signal, 0))
    attn = attn[0, :, 0, 1:]  # cls_token attention with rest
    attn = tf.transpose(attn, (1, 0))
    attn = tf.expand_dims(tf.expand_dims(attn, 0), 0)
    attn = tf.image.resize(attn, (1, 5000))[0, 0]

    for head in range(6):
        fig, ax = plt.subplots(figsize=(10, 1.5))
        plot_ax(ax, signal=record.p_signal[:, lead], plot_grid=False, sampling_rate=record.fs)
        ax.pcolorfast(ax.get_xlim(), ax.get_ylim(), attn[:, head][np.newaxis], cmap="Reds", alpha=0.6)
        ax.set_ylabel(lead_index[lead], fontsize=13)
        fig.savefig(f"attn_{head}")


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="path to file")
    parser.add_argument("weights", help="path to model")
    parser.add_argument("lead", help="lead index no")
    args = parser.parse_args()
    main(args.file, args.weights, int(args.lead))
