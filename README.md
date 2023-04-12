A simplified implementation of Neural Audio Fingerprint for High-Specific Audio Retrieval Based on Contrastive Learning (NAF) in PyTorch. [Paper](https://arxiv.org/abs/2010.11910)


To obtain the data, run the following commands:
```mkdir data
cd data
wget https://collect.qmul.ac.uk/down?t=5DSGJQ07VCPS2S4C/613IT8K160JFB9U6AILKEPO
unzip down\?t\=5DSGJQ07VCPS2S4C%2F613IT8K160JFB9U6AILKEPO
wget https://collect.qmul.ac.uk/down?t=65A2P9AQ9276OBA9/6L0T7NQ4ULSHHH13758JR6O
unzip down\?t\=65A2P9AQ9276OBA9%2F6L0T7NQ4ULSHHH13758JR6O
wget https://github.com/karoldvl/ESC-50/archive/master.zip
wget https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip

unzip master.zip
rm -rf master.zip
unzip down\?t\=4HTGJ1LPL5MGNJHE%2FR8RCNHTCDNH8IID6NB9RK6O
rm -rf down\?t\=4HTGJ1LPL5MGNJHE%2FR8RCNHTCDNH8IID6NB9RK6O
unzip Audio.zip
rm -rf Audio.zip
```

## Installation
`pip install -r requirements.txt`

## Training
`python scripts/train.py`

Save best checkpoint as `./best.ckpt`
## Run Identification
`python scripts/build_db.py` then `python scripts/audio_id.py`