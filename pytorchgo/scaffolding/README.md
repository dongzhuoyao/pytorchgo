# run rq

## compile redis and run 

```
./src/redis-server --protected-mode no
```

## install rq

```
pip install rq
```


## consume the tasks

```
rq worker exp0 --url redis://146.50.28.19:6379 
```

## debug your code in main.py 
```
python set_sweep_task -name exp0
```

## add 

add pytorchgo@pytorchgo.iam.gserviceaccount.com