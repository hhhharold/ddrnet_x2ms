import cv2
from model import get_seg_model
import os
import numpy as np
import time
import logging
import mindspore
import x2ms_adapter

logging.getLogger().setLevel(logging.INFO)
logging.info('start inference')
try:
    start_time = time.time()
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    img = 'test.png'
    img = cv2.imread(img)

    img = cv2.resize(img,(512,512))
    image = img.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std

    image = x2ms_adapter.tensor_api.transpose(image, (2,0,1))
    image = x2ms_adapter.from_numpy(image)
    image = x2ms_adapter.tensor_api.unsqueeze(image, 0)
    image = image

    model = get_seg_model()
    x2ms_adapter.x2ms_eval(model)
    infer_start_time = time.time()
    for i in range(50):
        out = model(image)
    end_time = time.time()
except Exception as e:
    logging.error(e)
logging.info('total_time: {}'.format(end_time - start_time))
logging.info('average_inference_time: {}'.format((end_time- infer_start_time)/50.0))