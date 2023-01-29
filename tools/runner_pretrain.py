import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils

train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudRotatePerturbation(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict

def generate_viewpoints(bs,dim=3):
    dim = dim
    radius = 1

    x = np.random.normal(0, 1, (bs, dim))

    z = np.linalg.norm(x, axis=1)
    z = z.reshape(-1, 1).repeat(x.shape[1], axis=1)

    Points = x / z * radius * np.sqrt(dim)

    return Points

def generate_raw_viewpoints(bs,vp,data):
    points = []
    for i in range(bs):
        idx = np.arange(data.size(1))
        np.random.shuffle(idx)
        idx = torch.tensor(idx[:vp])
        point = data[i,idx, ...]
        points.append(point.unsqueeze(0))

    points = torch.cat(points,dim=0)

    return points

def evaluate_svm(train_features, train_labels, test_features, test_labels):
    # clf = LinearSVC()
    clf = SVC(C=0.01, kernel='linear')
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]

def run_net(args, config, train_writer=None, val_writer=None, view_points=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)

    # train_dataloader_svm, test_dataloader_svm = builder.dataset_builder_modelnet40()

    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME


            if dataset_name == 'ShapeNet':
                if 'random' in args.vp:
                    n_vp = int(args.vp.split('_')[1])
                    if args.raw_block:
                        view_points_ = torch.tensor(generate_raw_viewpoints(data.size(0), n_vp, data))
                    else:
                        view_points_ = torch.tensor(generate_viewpoints(data.size(0) * n_vp)).view(data.size(0),n_vp, 3)
                else:
                    view_points_ = view_points.repeat(data.size(0), 1, 1)
                data = torch.cat([data, view_points_], dim=1)
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)
                assert points.size(1) == npoints
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')


            points = train_transforms(points)
            loss = base_model(points)
            try:
                loss.backward()
                # print("Using one GPU")
            except:
                loss = loss.mean()
                loss.backward()
                # print("Using multi GPUs")

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item()*1000])
            else:
                losses.update([loss.item()*1000])


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
             optimizer.param_groups[0]['lr']), logger = logger)

        # if epoch % args.val_freq == 0 and epoch != 0:
        #     # Validate the current model
        #     metrics = validate_modelnet40_svm(base_model, train_dataloader_svm, test_dataloader_svm, epoch, val_writer, args, config, logger=logger, view_points=view_points)
        #
        #     # Save ckeckpoints
        #     if metrics.better_than(best_metrics):
        #         best_metrics = metrics
        #         builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        if epoch % 25 ==0 and epoch >=250 or epoch==100 or epoch==200:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args,
                                    logger=logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate_modelnet40_svm(base_model, train_dataloader_svm, test_dataloader_svm, epoch, val_writer, args, config, logger = None, view_points=None):
    # print_log(f"[VALIDATION_ModelNet40] Start validating epoch {epoch}", logger=logger)
    feats_train = []
    labels_train = []
    base_model.eval()

    for i, (data, label) in enumerate(train_dataloader_svm):
        view_points_ = view_points.repeat(data.size(0), 1, 1)
        labels = list(map(lambda x: x[0], label.numpy().tolist()))
        data = torch.cat([data, view_points_], dim=1)
        data = data.cuda().contiguous()

        with torch.no_grad():
            feats = base_model(data, noaug=True)
        feats = feats.detach().cpu().numpy()
        for feat in feats:
            feats_train.append(feat)
        labels_train += labels

    feats_train = np.array(feats_train)
    labels_train = np.array(labels_train)

    feats_test = []
    labels_test = []

    for i, (data, label) in enumerate(test_dataloader_svm):
        view_points_ = view_points.repeat(data.size(0), 1, 1)
        labels = list(map(lambda x: x[0], label.numpy().tolist()))
        data = torch.cat([data, view_points_], dim=1)
        data = data.cuda().contiguous()
        with torch.no_grad():
            feats = base_model(data, noaug=True)
        feats = feats.detach().cpu().numpy()
        for feat in feats:
            feats_test.append(feat)
        labels_test += labels

    feats_test = np.array(feats_test)
    labels_test = np.array(labels_test)

    model_tl = SVC(C=0.01, kernel='linear')
    model_tl.fit(feats_train, labels_train)
    test_accuracy = model_tl.score(feats_test, labels_test)

    print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch,test_accuracy), logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', test_accuracy, epoch)

    return Acc_Metric(test_accuracy)

def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.dataset.train.others.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())


        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch,svm_acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)


def test_net():
    pass