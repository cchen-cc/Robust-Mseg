from PIL import Image
import numpy as np
import tensorflow as tf

# colour map
label_colours = [(0, 0, 0)
                 # 0=background
    , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    , (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0)
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    , (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128)
                 # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
    , (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]


# 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

def decode_labels(mask, num_images, num_classes):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w, d = mask.shape
    c = 1
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        label_tmp = mask[0, :, :, d // (num_images+1) * (i+1)]
        img = Image.new('RGB', (h, w))
        pixels = img.load()
        for j_, j in enumerate(label_tmp):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs


def decode_images(imgs, num_images):
    """Inverse preprocessing of the batch of images.
       Add the mean vector and convert from BGR to RGB.

    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.

    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    n, h, w, d, c = imgs.shape
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        output_tmp = imgs[0, :, :, d // (num_images+1)*(i+1), :]
        output_tmp = (output_tmp-output_tmp.min())/(output_tmp.max()-output_tmp.min())*255.0
        output_tmp = np.concatenate((output_tmp,output_tmp,output_tmp), axis=2)
        outputs[i] = output_tmp.astype(np.uint8)
    return outputs