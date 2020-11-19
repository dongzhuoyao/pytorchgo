# run rq

## compile redis and run 

```
./src/redis-server --protected-mode no
```

## install rq

```
pip install rq
```

```
python set_sweep_task -name exp0
```

```
rq worker exp0 --url redis://146.50.28.19:6379 
```