import argparse
import gspread

def parse_args(args):
  parser = argparse.ArgumentParser(description= 'few-shot script')
  parser.add_argument('-dataset', default='cars_default', help='')
  parser.add_argument('-save_xls',type=str)
  if len(args)==0:
      return parser.parse_args()
  else:
      return parser.parse_args(args)

def main(*args):
    args = parse_args(args)
    print(args)
    gc = gspread.service_account(filename="/home/thu/pytorchgo-817f6e741d1c.json")
    sh = gc.open("test_pytorchgo")
    sh.sheet1.update(args.save_xls,args.dataset)
    return args.dataset

if __name__ == '__main__':
  main(*('-dataset','cars'))
  main()


