# %cd /home/yan/score-sde/text2image/image_captioning/content
# !git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git
# %cd /home/yan/score-sde/text2image/image_captioning/content/vqa-maskrcnn-benchmark
# # Compile custom layers and build mask-rcnn backbone
# !python setup.py build
# !python setup.py develop

import sys
sys.path.append('/home/yan/score-sde/text2image/image_captioning/content/vqa-maskrcnn-benchmark')

import yaml
import cv2
import torch
import requests
import numpy as np
import gc
import torch.nn.functional as F
import pandas as pd
print(torch.__version__)  
print(torch.version.cuda) 
print(torch.cuda.is_available())

import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from IPython.display import display, HTML, clear_output
from ipywidgets import widgets, Layout
from io import BytesIO

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
torch._six.PY3 = True  # fix for newer pytorch
from maskrcnn_benchmark.utils.model_serialization import load_state_dict


class Interpolate(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size())
        image = F.interpolate(image, size)
        return image
    

class FeatureExtractor:
    TARGET_IMAGE_SIZE = [448, 448]
    CHANNEL_MEAN = [0.485, 0.456, 0.406]
    CHANNEL_STD = [0.229, 0.224, 0.225]
    
    def __init__(self):
        # self._init_processors()
        self.detection_model = self._build_detection_model()
    
#     def __call__(self, img):
#         # with torch.no_grad():
#         detectron_features = self.get_detectron_features(img)
        
#         return detectron_features

    def __call__(self, img):
        # with torch.no_grad():
        detectron_features = self.get_detectron_features(img)
        
        return detectron_features
    
    def _build_detection_model(self):

        cfg.merge_from_file('/home/yan/score-sde/text2image/image_captioning/content/model_data/detectron_model.yaml')
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load('/home/yan/score-sde/text2image/image_captioning/content/model_data/detectron_model.pth', 
                                                        map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        model.to("cuda")
        model.eval()
        return model

    def _image_transform(self, img):
        
        # im = np.array(img).astype(np.float32)
        # im = im[:, :, ::-1]
        # im -= np.array([102.9801, 115.9465, 122.7717])
        # im_shape = im.shape
        # im_size_min = np.min(im_shape[0:2])
        # im_size_max = np.max(im_shape[0:2])
        # im_scale = float(800) / float(im_size_min)
        # # Prevent the biggest axis from being more than max_size
        # if np.round(im_scale * im_size_max) > 1333:
        #          im_scale = float(1333) / float(im_size_max)
        # im = cv2.resize(
        #      im,
        #      None,
        #      None,
        #      fx=im_scale,
        #      fy=im_scale,
        #      interpolation=cv2.INTER_LINEAR
        #  )
        # img = torch.from_numpy(im).permute(2, 0, 1)

        img = torch.squeeze(img).permute(1, 2, 0)
        img_cp = img.clone()
        img = img.flip(dims=[-1])
        # img = img - torch.from_numpy(np.array([102.9801, 115.9465, 122.7717])).to("cuda")
        
        data_transform = transforms.Compose([ 
                    # transforms.Resize(256),                          # smaller edge of image resized to 256
                    # transforms.RandomCrop(224),                      # get 224x224 crop from random location
                    # transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
                    # transforms.Resize(800),
                    Interpolate(min_size=800, max_size=1333),
                    transforms.ToTensor(),                           # convert the PIL Image to a tensor
                    transforms.Normalize((102.9801, 115.9465, 122.7717),      # normalize image for pre-trained model
                                         (1, 1, 1))])
        
        img = img.permute(2, 0, 1)
        im_shape = img.size()
        print('im_shape: ', im_shape)
        im_size_min = torch.min(torch.tensor(im_shape[0:2]))
        im_size_max = torch.max(torch.tensor(im_shape[0:2]))
        im_scale = float(800) / float(im_size_min)
#         # Prevent the biggest axis from being more than max_size
        if torch.round(im_scale * im_size_max) > 1333:
            im_scale = float(1333) / float(im_size_max)
        
#         resize = transforms.Compose([
#             transforms.Resize(800)
#         ])  
#         img = resize(img)
        
        return img, im_scale
        # return torch.tensor(img, dtype=torch.float), im_scale


    def _process_feature_extraction(self, output,
                                    im_scales,
                                    feat_name='fc6',
                                    conf_thresh=0.2):
        batch_size = len(output[0]["proposals"])
        n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feat_name].split(n_boxes_per_image)
        cur_device = score_list[0].device

        feat_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            scores = score_list[i]

            max_conf = torch.zeros((scores.shape[0])).to(cur_device)

            for cls_ind in range(1, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                             cls_scores[keep], max_conf[keep])

            keep_boxes = torch.argsort(max_conf, descending=True)[:100]
            feat_list.append(feats[i][keep_boxes])
        return feat_list
        
    def get_detectron_features(self, img):
        img = img.to("cuda")
        im, im_scale = self._image_transform(img)
        img_tensor, im_scales = [im], [im_scale]
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        print("current_img_list: ", current_img_list.tensors)
        current_img_list = current_img_list.to('cuda')
        # with torch.no_grad():
        output = self.detection_model(current_img_list)
        feat_list = self._process_feature_extraction(output, im_scales, 'fc6', 0.2)
        return feat_list[0]
    

# def get_captions(img_feature):
#     # Return the 5 captions from beam serach with beam size 5
#     return model.decode_sequence(model(img_feature.mean(0)[None], img_feature[None], mode='sample', 
#                                        opt={'beam_size':5, 'sample_method':'beam_search', 'sample_n':1})[0])