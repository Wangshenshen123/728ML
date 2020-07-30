from datetime import datetime
import os
import numpy as np

import tensorflow as tf
# from tensorlayer.layers import Layer, Input, Dropout, Dense
from tensorlayer.models import Model

# from models import Decoder, Encoder
from modules.models.tensorlayer.vgg import vgg19_rev, vgg19

# from scipy.misc import imread, imsave
from utils import imread, imsave
import utils

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# TL1to2: self-defined VGG-alike models -> reuse pretrained models\vgg.py
# ENCODER_WEIGHTS_PATH = 'pretrained_models/pretrained_vgg19_encoder_weights.npz'
# DECODER_WEIGHTS_PATH = 'pretrained_models/pretrained_vgg19_decoder_weights.npz'
DEC_BEST_WEIGHTS_PATH = 'trained_models/dec_34700(loss=427986048.00)_weights.h5'

content_path = 'test_images/content/'
style_path = 'test_images/style/'
output_path = 'test_images/output/'

class StyleTransferModel(Model):
    def __init__(self, *args, **kwargs):
        super(StyleTransferModel, self).__init__(*args, **kwargs)
        # NOTE: you may use a vgg19 instance for both content encoder and style encoder, just as in train.py
        self.enc_c_net = vgg19(pretrained=True, end_with='conv4_1', name='content')
        self.enc_s_net = vgg19(pretrained=True, end_with='conv4_1', name='style')
        self.dec_net = vgg19_rev(pretrained=False, end_with='conv1_1', input_depth=512, name='stylized_dec')
        if os.path.exists(DEC_BEST_WEIGHTS_PATH):
            self.dec_net.load_weights(DEC_BEST_WEIGHTS_PATH)

    def forward(self, inputs, training=None, alpha=1):
        """
        :param inputs: [content_batch, style_batch], both have shape as [batch_size, w, h, c]
        :param training:
        :param alpha:
        :return:
        """
        # TL1to2: preprocessing and reverse -> vgg forward() will handle it
        # # switch RGB to BGR
        # content = tf.reverse(content_input, axis=[-1])
        # style = tf.reverse(style_input, axis=[-1])
        # # preprocess image
        # content = Encoder.preprocess(content_input)
        # style = Encoder.preprocess(style_input)
        content, style = inputs

        # encode image
        # we should initial global variables before restore model
        content_features = self.enc_c_net(content)
        style_features = self.enc_s_net(style)

        # pass the encoded images to AdaIN  # IMPROVE: try alpha gradients
        target_features = utils.AdaIN(content_features, style_features, alpha=alpha)

        # decode target features back to image
        generated_img = self.dec_net(target_features)

        # # deprocess image
        # generated_img = Encoder.reverse_preprocess(generated_img)
        # # switch BGR back to RGB
        # generated_img = tf.reverse(generated_img, axis=[-1])
        # # clip to 0..255
        # generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)

        return generated_img