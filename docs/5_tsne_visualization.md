# Trained Model Latent Space Visualization
Here, we provide guides for visualizing the trained autoencoder model's latent space.

### 1. Visualization
#### 1.1 Arguments
There are several arguments for running `src/run/tnse_test.py`:
* [`-r`, `--resume_model_dir`]: Directory to the model to visualize latent space. Provide the path up to `${project}/${name}`, and it will automatically select the model from `${project}/${name}/weights/` to visualize.
* [`-l`, `--load_model_type`]: Choose one of [`loss`, `last`].
    * `loss` (default): Resume the model with the minimum validation loss.
    * `last`: Resume the model saved at the last epoch.
* [`-d`, `--dataset_type`]: (default: `test`) Choose one of [`train`, `validation`, `test`].
* [`-n`, `--result_num`]: (default: `10`) The number of random data to visualize.


#### 1.2 Command
`src/run/tsne_test.py` file is used to visualize latent spaces of the trained model.
```bash
python3 src/run/tsne_test.py --resume_model_dir ${project}/${name}
```
