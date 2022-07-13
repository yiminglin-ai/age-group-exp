## Train on AWS slurm

```python
python train.py trainer.gpus=4 +trainer.strategy=ddp logger=wandb -m
```
