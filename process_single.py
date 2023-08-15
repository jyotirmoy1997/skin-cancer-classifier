import img_processing
import os

file = "ISIC_0024323"
path = f'D:\\Skin Cancer Project\\Demo\\Original\\mel\\{file}.jpg'
outpath = "D:\Output"

if not os.path.exists(outpath):
    os.makedirs(outpath)

img_processing.process(f'{path}',outpath, file)
