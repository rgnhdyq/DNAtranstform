# DNAtranstform
CUDA 50系 运行
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu124

  Dataset + k-mer Tokenizer：dataset.py
  Transformer 模型：model
  运行训练：train.py

  正常输出示例：
  Epoch 1 | Loss 0.4123 | AUC 0.78
  Epoch 2 | Loss 0.3318 | AUC 0.84
  ...
