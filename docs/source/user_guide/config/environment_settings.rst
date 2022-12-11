Environment settings
===========================
Environment settings are designed to set basic parameters of running environment.

- ``gpu_id (int or str)`` : The id of GPU device. Defaults to ``0``.
- ``use_gpu (bool)`` : Whether or not to use GPU. If True, using GPU, else using CPU.
  Defaults to ``True``.
- ``seed (int)`` : Random seed. Defaults to ``2020``.
- ``state (str)`` : Logging level. Defaults to ``'INFO'``.
  Range in ``['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL']``.
- ``encoding (str)``: Encoding to use for reading atomic files. Defaults to ``'utf-8'``.
  The available encoding can be found in `here <https://docs.python.org/3/library/codecs.html#standard-encodings>`__.
- ``reproducibility (bool)`` : If True, the tool will use deterministic
  convolution algorithms, which makes the result reproducible. If False,
  the tool will benchmark multiple convolution algorithms and select the fastest one,
  which makes the result not reproducible but can speed up model training in
  some case. Defaults to ``True``.
- ``data_path (str)`` : The path of input dataset. Defaults to ``'dataset/'``.
- ``checkpoint_dir (str)`` : The path to save checkpoint file.
  Defaults to ``'saved/'``.
- ``show_progress (bool)`` : Show the progress of training epoch and evaluate epoch.
  Defaults to ``True``.
- ``save_dataset (bool)``: Whether or not save filtered dataset.
  If True, save filtered dataset, otherwise it will not be saved.
  Defaults to ``False``.
- ``save_dataloaders (bool)``: Whether or not save split dataloaders.
  If True, save split dataloaders, otherwise they will not be saved.
  Defaults to ``False``.
