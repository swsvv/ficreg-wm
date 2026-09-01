import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

class ModelObjectCallBack(Callback):
    """Saves model checkpoints after each epoch and optionally stops training early."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1,
                 stop_epoch: int = None):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval
        self.stop_epoch = stop_epoch

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        epoch = trainer.current_epoch + 1
        output_path = self.dirpath / f"{self.filename}_epoch_{epoch}_object.ckpt"
        is_final = (epoch == trainer.max_epochs) or (
            self.stop_epoch is not None and epoch >= self.stop_epoch
        )

        if trainer.is_global_zero:
            if epoch % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            elif epoch == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

            if is_final:
                final_path = self.dirpath / f"{self.filename}_object.ckpt"
                self._dump_model(pl_module.model, final_path)

        # Signal Lightning to stop after this epoch completes cleanly.
        if self.stop_epoch is not None and epoch >= self.stop_epoch:
            trainer.should_stop = True

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")