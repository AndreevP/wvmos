# WV-MOS
MOS score prediction by fine-tuned wav2vec2.0 model

**Keywords:** MOS-Net, MB-Net, PESQ, STOI, speech quality

## Getting started
The package installation was tested with python3.9

```bash
pip install git+https://github.com/AndreevP/wvmos
```
## Inference

```python
from wvmos import get_wvmos
model = get_wvmos(cuda=True)

mos = model.calculate_one("path/to/wav/file") # infer MOS score for one audio 

mos = model.calculate_dir("path/to/dir/with/wav/files", mean=True) # infer average MOS score across .wav files in directory
```

## Citation and Acknowledgment

Coming soon
