LSTM Recurrent Neural Network for text generation
=================================================

Folders `data` and `models` must exist.

Build a model:
```
$ INPUT_TEXT=data/call_of_cthulhu.txt python build.py
```

Many files may be generated in `models`, you need to choose one of the
weights file for the generation process:
```
$ WEIGHT_FILE=models/weights-20-2.5038.hdf5 python generate.py
```
