# sequence-classification
Use GRU model, attention GRU model, google-bert model for imdb label dataset classification.

developed by pytorch with pytorch_pretrained_bert.

to install pytorch_pretrained_bert, use pip3 install pytorch_pretrained_bert.

imdb label dataset structure: a txt file with [strings label] structure.

you can modify your own model by changing the layers of model or other parameters, such as hidden dim or attention qkv dim.

the learning ratio for GRU and attention GRU model can be chosen as 0.01, however the bert model's learning ratio should better be choosed as 1e-5 or even less.
