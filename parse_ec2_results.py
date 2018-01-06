import imageio
from PIL import Image
import numpy as np

input_folder = './ec2_output_deconv/'
output_name = 'deconv'
output_size = 200

def make_gif(output_file=output_name+'.gif', folder=input_folder+'output/', output_size=output_size):
    imgs = []
    for i in range(100):
        img = Image.open('{}output_{}.jpg'.format(folder,i))
        img = img.resize((output_size, output_size))
        img = np.array(img)
        imgs.append(img)
    imageio.mimsave(output_file, imgs)

def img_parse(input_file, output_file, output_size=output_size):
    img = Image.open(input_file)
    img = img.resize((output_size, output_size))
    img.save(output_file)


if __name__=='__main__':
    make_gif()
    img_parse(input_folder+'deconstructed.jpg','./'+output_name+'_decon.jpg')
    img_parse(input_folder+'/output/output_100.jpg','./'+output_name+'_final.jpg')
    img_parse(input_folder+'bunny_512.jpg','./'+output_name+'_truth.jpg')

    
