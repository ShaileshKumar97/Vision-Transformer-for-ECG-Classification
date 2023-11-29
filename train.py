import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


from data import gen_datasets, gen_df
from vit import VisionTransformer


def train(data_path):

    # df = gen_df(Path(database))
    # train_ds, val_ds = gen_datasets(df, 64)

    # training dataset
    train_x = np.load(f"{data_path}/ecg_x_train.npy")
    train_y = np.load(f"{data_path}/ecg_y_train.npy")
    batch_size = 32
    shuffle_buffer_size = len(train_x)  # Set it to the number of samples for perfect shuffling

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_dataset = train_dataset.shuffle(buffer_size=shuffle_buffer_size)
    train_dataset = train_dataset.batch(batch_size)
    
    # validaton dataset
    val_x = np.load(f"{data_path}/ecg_x_test.npy")
    val_y = np.load(f"{data_path}/ecg_y_test.npy")
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
    val_dataset = val_dataset.batch(batch_size)

    epochs = 25
    vit = VisionTransformer(
        patch_size=20,
        hidden_size=768,
        depth=6,
        num_heads=6,
        mlp_dim=256,
        num_classes=5, # len(df["y"].values[0]),
        sd_survival_probability=0.9,
    )

    optimizer = tf.keras.optimizers.Adam(0.0001)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.AUC(from_logits=True, name="roc_auc")]
    vit.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            "/model-checkpoint/", monitor="val_roc_auc", save_best_only=True, save_weights_only=True
        )
    ]

    vit.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=cbs)
    vit.save("vit_ecg_model", save_format='tf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("database", help="root folder of database")

    args = parser.parse_args()
    train(args.database)
