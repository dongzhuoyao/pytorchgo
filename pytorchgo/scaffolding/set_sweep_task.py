from rq import Queue
from redis import Redis
import time
import sweep_all_job
import argparse

def parse_args():
  parser = argparse.ArgumentParser(description= 'few-shot script')
  parser.add_argument('-name', default='exp0')
  return parser.parse_args()

args = parse_args()

# Tell RQ what Redis connection to use
redis_conn = Redis()
q = Queue(name=args.name, connection=redis_conn)  # no args implies the default queue


job = q.enqueue(sweep_all_job.main, args=['-dataset','cars'],description='description')
print(job.result)   # => None

