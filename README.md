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
