To recreate issue:

```
pip install -r requirements.txt
python main.py
```

My results are (CPU):
```
Keras 100 predict iterations took 6.44 seconds
Keras: 64.44 ms per prediction
Torch 100 predict iterations took 0.05 seconds
Torch: 0.51 ms per prediction
```

Updated Results with variatons (CPU/m2)
```
Keras with .predict() in a loop
Keras 100 predict iterations took 2.14 seconds
Keras: 21.37 ms per prediction
---
Keras with __call__ in a loop:
Keras 100 predict iterations took 0.73 seconds
Keras: 7.33 ms per prediction
---
Keras with predict dataset:
Keras 100 predict iterations took 0.21 seconds
Keras: 2.09 ms per prediction
---
Torch with numpy in a loop:
Torch 100 predict iterations took 0.04 seconds
Torch: 0.41 ms per prediction
---
Torch with tensor in a loop:
Torch 100 predict iterations took 0.03 seconds
Torch: 0.25 ms per prediction
```