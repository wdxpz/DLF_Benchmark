import gzip
import numpy
import struct
import cv2
from six.moves import range

dst_img_size = 64

def reader_creator(image_filename, label_filename, buffer_size):
    def reader():
        with gzip.GzipFile(image_filename, 'rb') as image_file:
            img_buf = image_file.read()
            with gzip.GzipFile(label_filename, 'rb') as label_file:
                lab_buf = label_file.read()

                step_label = 0

                offset_img = 0
                # read from Big-endian
                # get file info from magic byte
                # image file : 16B
                magic_byte_img = '>IIII'
                magic_img, image_num, rows, cols = struct.unpack_from(
                    magic_byte_img, img_buf, offset_img)
                offset_img += struct.calcsize(magic_byte_img)

                offset_lab = 0
                # label file : 8B
                magic_byte_lab = '>II'
                magic_lab, label_num = struct.unpack_from(magic_byte_lab,
                                                          lab_buf, offset_lab)
                offset_lab += struct.calcsize(magic_byte_lab)

                while True:
                    if step_label >= label_num:
                        break
                    fmt_label = '>' + str(buffer_size) + 'B'
                    labels = struct.unpack_from(fmt_label, lab_buf, offset_lab)
                    offset_lab += struct.calcsize(fmt_label)
                    step_label += buffer_size

                    fmt_images = '>' + str(buffer_size * rows * cols) + 'B'
                    images_temp = struct.unpack_from(fmt_images, img_buf,
                                                     offset_img)
                    # images = numpy.reshape(images_temp, (
                    #     buffer_size, rows * cols)).astype('float32')
                    images = numpy.reshape(images_temp, (
                        buffer_size, rows * cols)).astype('float32')
                    offset_img += struct.calcsize(fmt_images)

                    
                    resized_images = numpy.zeros((buffer_size, dst_img_size * dst_img_size)).astype('float32')
                    print(resized_images.shape)
                    for i in range(buffer_size):
                        img = images[i].reshape((rows, cols))
                        resieze_img = cv2.resize(img, dsize=(dst_img_size, dst_img_size), interpolation=cv2.INTER_LINEAR)
                        resized_images[i] = resieze_img.flatten()
                        break

                    print(resized_images.shape)


                    images = images / 255.0 * 2.0 - 1.0
                    for i in range(buffer_size):
                        yield images[i, :], int(labels[i])

    return reader

def batch(reader, batch_size, drop_last=False):
    """
    Create a batched reader.

    :param reader: the data reader to read from.
    :type reader: callable
    :param batch_size: size of each mini-batch
    :type batch_size: int
    :param drop_last: drop the last batch, if the size of last batch is not equal to batch_size.
    :type drop_last: bool
    :return: the batched reader.
    :rtype: callable
    """

    def batch_reader():
        r = reader()
        b = []
        for instance in r:
            b.append(instance)
            if len(b) == batch_size:
                yield b
                b = []
        if drop_last == False and len(b) != 0:
            yield b

    # Batch size check
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size should be a positive integeral value, "
                         "but got batch_size={}".format(batch_size))

    return batch_reader