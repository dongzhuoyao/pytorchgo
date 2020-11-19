from rq import Queue
from redis import Redis
import time
import sweep_all_job

# Tell RQ what Redis connection to use
redis_conn = Redis()
q = Queue(connection=redis_conn)  # no args implies the default queue


job = q.enqueue(sweep_all_job.main, ['-dataset','cars'])
print(job.result)   # => None

