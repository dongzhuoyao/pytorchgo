import gspread

gc = gspread.service_account(filename="/home/thu/pytorchgo-817f6e741d1c.json")
sh = gc.open("test_pytorchgo")
print(sh.sheet1.get('A1'))