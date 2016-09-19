import PIL
from PIL import Image

W_final = 600
H_final = 60
new_size = (W_final, H_final)
new_im = Image.new("RGB", new_size, (255,255,255))   ## luckily, this is already black!

new_im.save('white.png')

