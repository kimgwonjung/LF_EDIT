import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import random
import numpy as np
import time
import shutil
import scipy.io

from enum import Enum
from PIL import Image

from collections import OrderedDict
from dataset import Dataset
from einops import rearrange
from loss import PerceptualLoss
from DPT.dpt.models import DPTDepthModel, CompositionNetwork
from network import Network

def make_config():
    parser = argparse.ArgumentParser()
    # Project
    parser.add_argument('--name', type=str, default="TEST1", help="train model name")
    parser.add_argument('--mode', type=int, default=0, help="0 is train, 1 is inference")
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)

    # DATA
    parser.add_argument('--train_path', type=str, default="./MI_TRAIN_TEST_ALL.csv")
    parser.add_argument('--val_path', type=str, default="./MI_TRAIN_TEST_ALL.csv")
    parser.add_argument('--inference_path', type=str, default="./")
    parser.add_argument('--output_path', type=str, default="./Result")
    parser.add_argument('--model_path', type=str, default="ckpt.pt")
    parser.add_argument('--sample_interval', type=int, default=3)
    parser.add_argument('--gt_prefix', type=str, default='/workspace/ManagerOffice_bl1_fordemo/GT/')
    parser.add_argument('--back_prefix', type=str, default='/workspace/ManagerOffice_bl1_fordemo/BACK/')
    parser.add_argument('--disp_prefix', type=str, default='/workspace/ManagerOffice_bl1_fordemo/BACK/')
    
    parser.add_argument('--resume_model_path', type=str, default='./model.pth.tar')
    parser.add_argument('--print_freq', type=int, default=5)

    parser.add_argument('--gt_key', type=str, default="gt")
    parser.add_argument('--back_key', type=str, default='back')

    # GPU & CPU
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--gpu', type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dist-url', type=str, default="tcp://127.0.0.1:7777")
    parser.add_argument('--dist-backend', type=str, default="nccl")
    parser.add_argument('--multiprocessing_distributed', default=True)

    # Hyper Parameter
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--img_size', type=int, default=256)

    parser.add_argument('--Perceptual_Weight', type=float, default=0.1)
    parser.add_argument('--Gradient_Weight', type=float, default=2.)
    parser.add_argument('--L1_Weight', type=float, default=1.)

    args = parser.parse_args()
    return args


def main():
    args = make_config()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


best_acc1 = 99999


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1

    if args.gpu is not None:
        print("Use GPU: {} for training".format(gpu))

    if args.distributed:
        if args.dist_url == "envs://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    #model = Network(imgh=384,imgw=384, patch_size=12, embed_dim=432)
    model = CompositionNetwork()
    
    
    state_dict = torch.load(args.resume_model_path, map_location=torch.device('cpu'))['state_dict']
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')

    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(gpu)
            model.cuda(gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    # context_perc_loss = Context_Perc_Loss().cuda(gpu)
    perc_loss = PerceptualLoss().cuda(gpu)
    l1_loss = nn.L1Loss().cuda(gpu)

    torch.backends.cudnn.benchmark = True

    train_dataset = Dataset(args, args.train_path)
    val_dataset = Dataset(args, args.val_path)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    data_length = len(train_loader)
    optimizer = optim.Adam(model.parameters(), args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=data_length / (args.batch_size/ngpus_per_node), eta_min=0.0000001)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                print("Using GPU")
                loc = 'cuda:{}'.format(gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    dist.barrier()
    for epoch in range(args.epoch):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train
        train(train_loader, model, [l1_loss, perc_loss], optimizer, epoch, args, gpu)

        # validation
        valiation(val_loader, model, [l1_loss, perc_loss], epoch, args, gpu)

        dist.barrier()
        scheduler.step()

        dist.barrier()

        # is_best = v_total_loss < best_acc1
        # best_acc1 = min(v_total_loss, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and gpu == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                # 'best_acc1' : best_acc1,
                # 'optimizer' : optimizer.state_dict(),
                # 'scheduler' : scheduler.state_dict()
            }, True, epoch + 1)


def train(train_loader, model, criterion, optimizer, epoch, args, gpu):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    perc_loss = AverageMeter('P_Loss', ':.4e', Summary.NONE)
    l1_loss = AverageMeter('L1_Loss', ':.4e', Summary.NONE)

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, l1_loss, perc_loss],
        prefix="Epoch: [{}]".format(epoch)
    )

    mde_model = DPTDepthModel(path="../model/dpt_hybrid-midas-501f0c75.pt", backbone="vitb_rn50_384").cuda()
    mde_model.eval()
    L1_loss, Perc_loss = criterion
    model.train()
    end = time.time()

    for idx, items in enumerate(train_loader):
        GT, BACK, DISP = items[0].cuda(gpu, non_blocking=True), items[1].cuda(gpu, non_blocking=True), items[2].cuda(
            gpu, non_blocking=True)
        data_time.update(time.time() - end)
        Output = torch.zeros_like(GT).cuda(gpu, non_blocking=True)
        #Visit = torch.zeors((GT.size(0),7,7)).cuda(gpu, non_blocking=True)
        #Visit = [[False for _ in range(7)] for _ in range(7)]
        Output[:,:,3,3] = GT[:,:,3,3].clone()

        cnt = 0
        
        if epoch < 2:
            angular_range = [[3, 3]]
        elif epoch >= 2 and epoch < 3:
            angular_range = [[3, 3], [2, 2], [2, 4], [4, 2], [4, 4]]
        else:
            angular_range = [[3, 3], [2, 2], [2, 4], [4, 2], [4, 4], [1, 1], [1, 3], [1, 5], [3, 1], [3, 5], [5, 1], [5, 3], [5, 5]]
        
        
        #angular_range = [[3, 3], [2, 2], [2, 4], [4, 2], [4, 4], [1, 1], [1, 3], [1, 5], [3, 1], [3, 5], [5, 1], [5, 3], [5, 5]]
        for angular_index in angular_range:
            target_img = Output[:,:,angular_index[0], angular_index[1]]

            with torch.no_grad():
                disp = mde_model(target_img)
                n_disp = disparity_norm_Real(disp, DISP).detach()

            target_LF = propagation(target_img, n_disp)
            gt_LF = GT[:,:,angular_index[0]-1:angular_index[0]+2,angular_index[1]-1:angular_index[1]+2].clone()
            back_LF = BACK[:,:,angular_index[0]-1:angular_index[0]+2,angular_index[1]-1:angular_index[1]+2].clone()
            IMG_L1_Loss = 0.0
            IMG_Perc_Loss = 0.0

            for s_y in range(0, 512, 128):
                for s_x in range(0, 512, 128):
                    crop_target_LF = target_LF[:,:,:,:,s_y:s_y+128, s_x:s_x+128]
                    crop_back_LF = back_LF[:,:,:,:,s_y:s_y+128, s_x:s_x+128]
                    crop_target_MI = rearrange(crop_target_LF, "B C V U H W -> B C (H V) (W U)")
                    crop_back_MI =  rearrange(crop_back_LF, "B C V U H W -> B C (H V) (W U)")
                    # Network

                    cnt += 1
                    crop_intermediate_MI_Output = model(crop_target_MI, crop_back_MI)
                    crop_intermediate_LF_Output = rearrange(crop_intermediate_MI_Output, "B C (H V) (W U) -> B C V U H W", V=3, U=3)

                    L_loss_1 = 0
                    P_loss_1 = 0

                    for v in range(3):
                        for u in range(3):
                            L_loss_1 += L1_loss(crop_intermediate_LF_Output[:,:,v,u], gt_LF[:,:,v, u, s_y:s_y+128, s_x:s_x+128])
                            P_loss_1 += Perc_loss(crop_intermediate_LF_Output[:,:,v,u], gt_LF[:,:,v, u, s_y:s_y+128, s_x:s_x+128]) * args.Perceptual_Weight


                    Output[:, :, angular_index[0] - 1:angular_index[0] + 2, angular_index[1] - 1:angular_index[1] + 2, s_y:s_y + 128, s_x:s_x + 128]  = crop_intermediate_LF_Output.detach()

                    IMG_L1_Loss += L_loss_1
                    IMG_Perc_Loss += P_loss_1
                    Total_loss = L_loss_1 + P_loss_1

                    optimizer.zero_grad()
                    Total_loss.backward()
                    optimizer.step()




        IMG_L1_Loss /= cnt
        IMG_Perc_Loss /= cnt
        perc_loss.update(IMG_L1_Loss.item(), GT.size(0))
        l1_loss.update(IMG_Perc_Loss.item(), GT.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if idx % args.print_freq == 0:
            progress.display(idx)




def valiation(val_loader, model, criterion, epoch, args, gpu):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    perc_loss = AverageMeter('P_Loss', ':.4e', Summary.NONE)
    l1_loss = AverageMeter('L1_Loss', ':.4e', Summary.NONE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, l1_loss, perc_loss],
        prefix='Test: '
    )

    mde_model = DPTDepthModel(path="../model/dpt_hybrid-midas-501f0c75.pt", backbone="vitb_rn50_384").cuda()
    mde_model.eval()

    L1_loss, Perc_loss = criterion

    model.eval()
    epoch_total_loss = 0

    with torch.no_grad():
        end = time.time()
        for idx, items in enumerate(val_loader):
            GT, BACK, DISP = items[0].cuda(gpu, non_blocking=True), items[1].cuda(gpu, non_blocking=True), items[2].cuda(gpu, non_blocking=True)
            Output = torch.zeros_like(GT).cuda(gpu, non_blocking=True)
            Visit = [[False for _ in range(7)] for _ in range(7)]
            Output[:, :, 3, 3] = GT[:, :, 3, 3].clone()
            cnt = 0
            '''
            if epoch < 3:
                angular_range = [[3, 3]]

            elif epoch >= 3 and epoch < 7:
                angular_range = [[3, 3], [2, 2], [2, 4], [4, 2], [4, 4]]
            else:
                angular_range = [[3, 3], [2, 2], [2, 4], [4, 2], [4, 4], [1, 1], [1, 3], [1, 5], [3, 1], [3, 5], [5, 1],
                                 [5, 3], [5, 5]]
            '''
            angular_range = [[3, 3], [2, 2], [2, 4], [4, 2], [4, 4], [1, 1], [1, 3], [1, 5], [3, 1], [3, 5], [5, 1], [5, 3], [5, 5]]
            for angular_index in angular_range:
                target_img = Output[:, :, angular_index[0], angular_index[1]]
                with torch.no_grad():
                    disp = mde_model(target_img)
                    n_disp = disparity_norm_Real(disp, DISP).detach()
                target_LF = propagation(target_img, n_disp)
                back_LF = BACK[:, :, angular_index[0] - 1:angular_index[0] + 2, angular_index[1] - 1:angular_index[1] + 2]

                for s_y in range(0, 512, 128):
                    for s_x in range(0, 512, 128):
                        crop_target_LF = target_LF[:, :, :, :, s_y:s_y + 128, s_x:s_x + 128]
                        crop_back_LF = back_LF[:, :, :, :, s_y:s_y + 128, s_x:s_x + 128]
                        crop_target_MI = rearrange(crop_target_LF, "B C V U H W -> B C (H V) (W U)")
                        crop_back_MI = rearrange(crop_back_LF, "B C V U H W -> B C (H V) (W U)")

                        # Network
                        cnt += 1
                        crop_intermediate_MI_Output = model(crop_target_MI, crop_back_MI)
                        crop_intermediate_LF_Output = rearrange(crop_intermediate_MI_Output, "B C (H V) (W U) -> B C V U H W", V=3, U=3)

                        Output[:, :, angular_index[0] - 1:angular_index[0] + 2, angular_index[1] - 1:angular_index[1] + 2, s_y:s_y + 128, s_x:s_x + 128] = crop_intermediate_LF_Output

            # L_loss = L1_loss(Intermediate_Output[:, :, 2:5, 2:5], GT[:, :, 2:5, 2:5])
            # epoch_total_loss += L_loss
            if idx % args.print_freq == 0:
                progress.display(idx)

            if idx % args.sample_interval == 0 and gpu == 0:
                if not os.path.exists(args.output_path):
                    os.makedirs(args.output_path)
                iter_path = os.path.join(args.output_path, "VAL_" + str(idx).zfill(4))
                if not os.path.exists(iter_path):
                    os.makedirs(iter_path)
                save_SAIs(GT, iter_path, "GT")
                save_SAIs(BACK, iter_path, "BACK")
                save_SAIs(Output, iter_path, "Intermediate")
                #save_SAIs(Output, iter_path, "INF")
                #save_SAIs(Mask, iter_path, "MASK")

    # return epoch_total_loss / len(val_loader)


def propagation(target, target_disp):
    b,c,h,w = target.shape

    temp = torch.zeros((b, c, 3, 3, h, w)).cuda()
    x = torch.linspace(0, w-1, steps=w)
    y = torch.linspace(0, h-1, steps=h)

    grid_y, grid_x = torch.meshgrid(y, x)

    grid_y = grid_y.expand(b, h, w).cuda()
    grid_x = grid_x.expand(b, h, w).cuda()

    dir_v = [1,0,-1]
    dir_u = [1,0,-1]
    for i, v in enumerate(dir_v):
        for ii, u in enumerate(dir_u):
            grid_ny = (((grid_y - v * target_disp) / h) - 0.5) * 2
            grid_nx = (((grid_x - u * target_disp) / w) - 0.5) * 2
            index_yx = torch.stack([grid_nx, grid_ny], -1).squeeze(1).float()
            temp[:,:,i,ii] = F.grid_sample(target, index_yx, padding_mode="border", align_corners=True)

    return temp

def disparity_norm(target, ori_disp):
    _max, _min = target.max(), target.min()
    scale = ori_disp.max()
    return ((target - _min) / (_max - _min)) * scale

def disparity_norm_Real(target, ori_disp):
    _max, _min = target.max(), target.min()
    ori_max, ori_min = ori_disp.max(), ori_disp.min()
    scale = ori_max + abs(ori_min)
    n_disp = ((target - _min) / (_max - _min)) * scale
    n_disp -= abs(ori_min)
    return n_disp

def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_{}.pth.tar'.format(str(epoch)))


def save_img(img, path, name):
    img = img.permute(0, 2, 3, 1) * 255.0
    im = Image.fromarray(img[0].cpu().numpy().astype(np.uint8).squeeze())
    im.save(os.path.join(path, name + ".png"))


def save_SAIs(img, path, name):
    img = img.permute(0, 2, 3, 4, 5, 1) * 255.0
    for v in range(7):
        for u in range(7):
            im = Image.fromarray(img[0, v, u].cpu().numpy().astype(np.uint8).squeeze())
            im.save(os.path.join(path, name + "_{}_{}.png".format(str(v), str(u))))


def save_mat(disp, path, name):
    scipy.io.savemat(os.path.join(path, name + ".mat"), {"disp": disp[0].cpu().numpy().squeeze()})


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()