from rq import Queue
from redis import Redis
import time
from sweep_all_job import main
import argparse

def parse_args():
  parser = argparse.ArgumentParser(description= 'few-shot script')
  parser.add_argument('-name', default='exp0')
  return parser.parse_args()

args = parse_args()

# Tell RQ what Redis connection to use
redis_conn = Redis()

q = Queue(name=args.name, connection=redis_conn)  # no args implies the default queue


job = q.enqueue(main, args=('-dataset','cars','-save_xls','A1'))
job = q.enqueue(main, args=('-dataset','cars1','-save_xls','A2'))
job = q.enqueue(main, args=('-dataset','cars2','-save_xls','A3'))
job = q.enqueue(main, args=('-dataset','cars3','-save_xls','A4'))

#print(job.result)   # => None

