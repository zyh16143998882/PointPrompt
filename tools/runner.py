import torch
import torch.nn as nn
import os
import json
from tools import builder
from tools.runner_pretrain import generate_raw_viewpoints, generate_viewpoints
from utils import misc, dist_utils
import time

from utils.AverageMeter import AverageMeter
from utils.logger import *

import cv2
import numpy as np


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    # base_model.load_model_from_ckpt(args.ckpts)
    builder.load_model(base_model, args.ckpts, logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)


# visualization
def test(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode
    target = './vis'
    # useful_cate = [
    #     "02691156", #plane
    #     "04379243",  #table
    #     "03790512", #motorbike
    #     "03948459", #pistol
    #     "03642806", #laptop
    #     "03467517",     #guitar
    #     "03261776", #earphone
    #     "03001627", #chair
    #     "02958343", #car
    #     "04090263", #rifle
    #     "03759954", # microphone
    # ]

    useful_cate = [
        "02691156",  # plane
    ]

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            print(idx)
            # import pdb; pdb.set_trace()
            if  taxonomy_ids[0] not in useful_cate:
                continue
            if taxonomy_ids[0] == "02691156":
                a, b= 90, 135
            elif taxonomy_ids[0] == "04379243":
                a, b = 30, 30
            elif taxonomy_ids[0] == "03642806":
                a, b = 30, -45

            elif taxonomy_ids[0] == "03467517":
                a, b = 0, 90
            elif taxonomy_ids[0] == "03261776":
                a, b = 0, 75
            elif taxonomy_ids[0] == "03001627":
                a, b = 30, -45
            else:
                a, b = 0, 0


            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            # dense_points, vis_points = base_model(points, vis=True)
            view_points_ = None
            if 'random' in args.vp:
                n_vp = int(args.vp.split('_')[1])
                if args.raw_block:
                    view_points_ = torch.tensor(generate_raw_viewpoints(points.size(0), n_vp, data))
                else:
                    view_points_ = torch.tensor(generate_viewpoints(points.size(0) * n_vp)).view(points.size(0), n_vp,3)
            data = torch.cat([data, view_points_], dim=1)
            points = data.cuda()
            dense_points, vis_points, full_gt, centers, vis_centers, loss1 = base_model(points, vis=True)

            view_points_back = view_points_.cuda()
            dense_points_back = dense_points
            vis_points_back = vis_points
            full_gt_back = full_gt
            centers_back = centers
            vis_centers_back = vis_centers
            for i in range(dense_points.size(0)):
                view_points = view_points_back[:,i,:]
                dense_points = dense_points_back[i]
                vis_points = vis_points_back[i]
                full_gt = full_gt_back[i]
                centers = centers_back[i]
                vis_centers = vis_centers_back[i]
                final_image = []
                data_path = f'./vis/{taxonomy_ids[0]}_{idx}'
                if not os.path.exists(data_path):
                    os.makedirs(data_path)

                points_ = points[:,:1024,:].squeeze().detach().cpu().numpy()
                np.savetxt(os.path.join(data_path,f'gt.txt'), points_, delimiter=';')
                points_ = misc.get_ptcloud_img(points_,a,b)       # 这里是图片了
                final_image.append(points_[150:650,150:675,:])

                view_points = view_points.squeeze().unsqueeze(0).detach().cpu().numpy()
                np.savetxt(os.path.join(data_path, f'view_points_{i}.txt'), view_points, delimiter=';')

                centers = centers.squeeze().detach().cpu().numpy()
                np.savetxt(os.path.join(data_path,f'center_{i}.txt'), centers, delimiter=';')

                vis_centers = vis_centers.squeeze().detach().cpu().numpy()
                np.savetxt(os.path.join(data_path, f'vis_center_{i}.txt'), vis_centers, delimiter=';')


                vis_points = vis_points.squeeze().detach().cpu().numpy()
                np.savetxt(os.path.join(data_path, f'vis_{i}.txt'), vis_points, delimiter=';')
                vis_points = misc.get_ptcloud_img(vis_points,a,b)

                final_image.append(vis_points[150:650,150:675,:])

                dense_points = dense_points.squeeze().detach().cpu().numpy()
                np.savetxt(os.path.join(data_path,f'dense_points_{i}.txt'), dense_points, delimiter=';')
                dense_points = misc.get_ptcloud_img(dense_points,a,b)
                final_image.append(dense_points[150:650,150:675,:])

                full_gt = full_gt.squeeze().detach().cpu().numpy()
                np.savetxt(os.path.join(data_path, f'full_gt_{i}.txt'), full_gt, delimiter=';')

                img = np.concatenate(final_image, axis=1)
                img_path = os.path.join(data_path, f'plot_{i}.jpg')
                cv2.imwrite(img_path, img)

            if idx > 1500:
                break

        return


def test_all(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode
    losses = AverageMeter(['Loss'])



    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            # import pdb; pdb.set_trace()
            print(idx)
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            # dense_points, vis_points = base_model(points, vis=True)
            view_points_ = None
            if 'random' in args.vp:
                n_vp = int(args.vp.split('_')[1])
                if args.raw_block:
                    view_points_ = torch.tensor(generate_raw_viewpoints(points.size(0), n_vp, data))
                else:
                    view_points_ = torch.tensor(generate_viewpoints(points.size(0) * n_vp)).view(points.size(0), n_vp,3)
            data = torch.cat([data, view_points_], dim=1)
            points = data.cuda()
            dense_points, vis_points, full_gt, centers, vis_centers, loss1 = base_model(points, vis=True)
            loss1 = loss1.mean()
            losses.update([loss1.item() * 1000])


        print_log('[Loss = %s' % ['%.4f' % l for l in losses.avg()], logger=logger)

        return