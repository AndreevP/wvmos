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

```
@misc{https://doi.org/10.48550/arxiv.2203.13086,
  doi = {10.48550/ARXIV.2203.13086},
  
  url = {https://arxiv.org/abs/2203.13086},
  
  author = {Andreev, Pavel and Alanov, Aibek and Ivanov, Oleg and Vetrov, Dmitry},
  
  keywords = {Sound (cs.SD), Machine Learning (cs.LG), Audio and Speech Processing (eess.AS), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  
  title = {HiFi++: a Unified Framework for Bandwidth Extension and Speech Enhancement},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

