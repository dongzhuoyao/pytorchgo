import argparse

def parse_args(args):
  parser = argparse.ArgumentParser(description= 'few-shot script')
  parser.add_argument('-dataset', default='cars', choices=['tieredimagenet','miniImagenet','cub','cars','places','plantae'], help='miniImagenet/cub/cars/places/plantae, specify multi for training with multiple domains')
  return parser.parse_args(args)

def main(args):
    if args is None:
        args = parse_args()
    else:
        args = parse_args(args)
    print(args)

if __name__ == '__main__':
  main(args=None)


