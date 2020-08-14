from keras.applications import inception_v3
from keras import backend as K

K.set_learning_phase(0)
"""===AidFunction========================================================="""
import scipy
from scipy.misc import imsave
from keras.preprocessing import image

def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)

def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)

def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x
"""======================================================================="""
model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
#model.summary()

layer_contributions = {
    'mixed0' : 0.2,
    'mixed1' : 0.4,
    'mixed2' : 0.6,
    'mixed3' : 1.8,
    #'mixed4' : 1.0,
    #'mixed5' : 1.2,
    #'mixed6' : 1.4,
    #'mixed7' : 1.6,
    #'mixed8' : 1.8,
    #'mixed9' : 2.,
    #'mixed10' : 2.2,
    }

layer_dict = dict([(layer.name, layer) for layer in model.layers])

loss = K.variable(0.)
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output

    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    loss = loss + coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling
    #loss = loss + coeff * K.sum(K.square(activation[:, :, :, :])) / scaling
    print(activation.shape)

dream = model.input

grads = K.gradients(loss, dream)[0]
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grads_values = outs[1]
    return loss_value, grads_values

def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grads_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grads_values
    return x
"""===DeepDreamSub========================================================"""
import numpy as np
step = 0.01
num_octave = 3
octave_scale = 1.4
iterations = 40

max_loss = 10.

base_image_path = r'D:\workspace\PyPro\0_Samples_DeepLearningWithPython\code_8\DeepDream.jpg'
img = preprocess_image(base_image_path)

original_shape = img.shape[1: 3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[: : -1]

original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img, iterations=iterations,
                          step=step, max_loss=max_loss)
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='dream_scale_' + str(shape) + '.png')

save_img(img, fname='final_dream.png')
