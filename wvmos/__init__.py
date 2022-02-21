from .wv_mos import Wav2Vec2MOS
import os
import urllib.request

path = os.path.join(os.path.expanduser('~'), ".cache/wv_mos/wv_mos.ckpt")

if (not os.path.exists(path)):
    print("Downloading the checkpoint for WV-MOS")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(
        "https://zenodo.org/record/6201162/files/wav2vec2.ckpt?download=1",
        path
    )
    print('Weights downloaded in: {} Size: {}'.format(path, os.path.getsize(path)))
    
def get_wvmos(cuda=True):
    return Wav2Vec2MOS(path, cuda=cuda)