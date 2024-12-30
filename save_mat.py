import cv2
import os
import numpy as np
import scipy.io as io

vis_path = 'test_images/TNO'
ir_path = 'test_images/TNO'
out_path = 'test_images/TNO/mat/'
ori_path = vis_path
if not os.path.exists(out_path):
    os.makedirs(out_path)
ls = os.listdir(ori_path)
d_path = []
ir_path = []
vis_path = []
for i in range(30):
    ir_name = os.path.join(ori_path, 'IR%d.png' % (i+1))
    vis_name = os.path.join(ori_path, 'VIS%d.png' % (i+1))
    ir = cv2.imread(ir_name, 0)
    vis = cv2.imread(vis_name, 0)
    h, w = ir.shape
    cube = np.zeros((2, h, w))
    cube[0, :, :] = ir
    cube[1, :, :] = vis
    io.savemat(out_path + '%d.mat' % (i+1), {'data':cube})