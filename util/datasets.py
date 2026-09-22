import torch

from .shapenet import ShapeNet

class AxisScaling(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter
        
    def __call__(self, surface, point):
        scaling = torch.rand(1, 3) * self.interval[0] + self.interval[1]
        surface = surface * scaling
        point = point * scaling

        scale = (1 / torch.abs(surface).max().item()) * 0.999999
        surface *= scale
        point *= scale

        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)

        return surface, point

def build_shape_surface_occupancy_dataset(split, args):
    if split == 'train':
        # transform = #transforms.Compose([
        transform = AxisScaling((0.75, 1.25), True)
        # ])
        return ShapeNet(args.mesh_folder, args.point_folder,
                        split=split, transform=transform, sampling=True, num_samples=256, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
    elif split == 'val':
        # return ShapeNet(args.data_path, split=split, transform=None, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
        return ShapeNet(args.mesh_folder, args.point_folder,
                        split=split, transform=None, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
    else:
        return ShapeNet(args.mesh_folder, args.point_folder,
                        split=split, transform=None, sampling=True, num_samples=1024, return_surface=True, surface_sampling=True, pc_size=args.point_cloud_size)
