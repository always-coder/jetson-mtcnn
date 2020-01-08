from facenet_pytorch import MTCNN
import torch
import numpy as np
from PIL import Image, ImageDraw


def show_bboxes(image, bounding_boxes):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for b in bounding_boxes:
        draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline='red')
    
    return img

def main():
    device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    image = Image.open('test.jpg')
    mtcnn = MTCNN(keep_all = True, device=device)
    bounding_boxes, _ = mtcnn.detect(image)

    img = show_bboxes(image, bounding_boxes)
    img.show()

if __name__ == '__main__':
    main()
