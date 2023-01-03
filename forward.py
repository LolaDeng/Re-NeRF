
from tqdm.auto import tqdm
from renderer import *
from utils import *
import sys

# modified from TensoRF https://github.com/apchenstu/TensoRF/blob/main/train.py

renderer = OctreeRender_trilinear_fast

def forward(args,stage,ckpt_path, dataset,masks =None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    white_bg = dataset.white_bg
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list

    # init parameters
    aabb = dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))

    # load model
    ckpt = torch.load(ckpt_path, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device':device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)
    

    # init optimiser
    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)
    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))
    
    if stage == "finetune":
        PSNRs = []

    allrays, allrgbs = dataset.all_rays, dataset.all_rgbs
    torch.cuda.empty_cache()
    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays.to(device), allrgbs.to(device), bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)
    
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    print(f"Re:NeRF == starting {stage} process ==")
    for iteration in pbar:
        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)
        rgb_map, alphas_map, depth_map, weights, uncertainty = renderer(rays_train, tensorf, chunk=args.batch_size,
                                N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)
        loss = torch.mean((rgb_map - rgb_train) ** 2)

        # loss 
        optimizer.zero_grad()
        loss.backward()
        loss = loss.detach().item()
        
        if stage == "finetune":
            optimizer.step()
            PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_factor

            # Print the current values of the losses.
            if iteration % args.progress_refresh_rate == 0:
                pbar.set_description(
                    f'Iteration {iteration:05d}:'
                    + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                    + f' mse = {loss:.6f}'
                )
                PSNRs = []

            if iteration == 0:
                n_voxels = args.N_voxel_final
                reso_cur = N_to_reso(n_voxels, tensorf.aabb)
                nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
                tensorf.upsample_volume_grid(reso_cur)

                if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                    # filter rays outside the bbox
                    allrays,allrgbs = tensorf.filtering_rays(allrays,allrgbs)
                    trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)

            if iteration in upsamp_list:
                if args.lr_upsample_reset:
                    print("reset lr to initial")
                    lr_scale = 1 #0.1 ** (iteration / args.n_iters)
                else:
                    lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
                grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
                optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
            
            # apply mask 
            with torch.no_grad():
                for module_name, module in tensorf.named_parameters():
                    if module_name in masks.keys():
                        module.data.mul_(masks[module_name])
    return tensorf

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]