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
rq worker --url redis://146.50.28.19:6379
```