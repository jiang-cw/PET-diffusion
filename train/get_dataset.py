from dataset import MRNetDataset, BRATSDataset, ADNIDataset, DUKEDataset, LIDCDataset, DEFAULTDataset
from torch.utils.data import WeightedRandomSampler


def get_dataset(cfg):
    # train_list = create_list(cfg.dataset.train_dir)
    # val_list = create_list(cfg.dataset.val_dir)
    # trainTransforms = [
    #         # NiftiDataset.Resample(opt.new_resolution, opt.resample),
    #         # NiftiDataset.Augmentation(),
    #         # NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
    #         NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
    #         ]

    # train_set = NifitDataSet(train_list, direction=opt.direction, transforms=trainTransforms, train=True)    # define the dataset and loader
    # train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)  # Here are then fed to the network with a defined batch size

    # valTransforms = [
    #     # NiftiDataset.Resample(opt.new_resolution, opt.resample),
    #     # NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
    #     # NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
    # ]

    # val_set = NifitDataSet(val_list, direction=opt.direction, transforms=valTransforms, test=True)
    # val_loader = DataLoader(val_set, batch_size= 1, shuffle=False, num_workers=opt.workers)
    train_dataset = DEFAULTDataset(
        root_dir=cfg.dataset.root_dir)
    val_dataset = DEFAULTDataset(
        root_dir=cfg.dataset.root_dir)
    sampler = None
    return train_loader, val_loader, sampler
   