
import os, copy, subprocess
from opt import config_parser
from tools_pruning.forward import forward
import  shutil
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from renderer import *
from utils import *
from dataLoader import dataset_dict
import torch.nn.utils.prune as prune
from tools_pruning import prune, forward, quantize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = OctreeRender_trilinear_fast
 
def run_pruning(args, val_dataset, pretrained,prune_percentage):
    stage = "prune"
    model = forward.forward(args,stage,pretrained, val_dataset)
    results = prune.pruning(model,target_layers,prune_percentage)
    return results 

def run_finetuning(args, train_dataset,ckpt_path, masks):
    stage = "finetune"
    model = forward.forward(args,stage, ckpt_path, train_dataset,masks)
    return model

def compress(path):
    subprocess.call(['7z', 'a', str(path) + ".7z", path]) 

if __name__ == '__main__':

    # manual seed 
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    # load data
    args = config_parser()
    dataset = dataset_dict[args.dataset_name]
    val_dataset = dataset(args.datadir, split='val', downsample=args.downsample_train, is_stack=False)
    train_dataset = dataset(args.datadir, split='train',
                            downsample=args.downsample_train, is_stack=False)
    
    # init 
    target_layers = ["density_plane","density_line","app_plane","app_line"]
    logfolder, fname = os.path.split(args.ckpt) 

    best_psnr = 0.0
    start = 1
    max_iteration = 1
    prune_per_step = 1/2

    for i in range(start, max_iteration +1): 

        prune_percentage = 1.0 - (prune_per_step)**i 
        prune_percentage = round(prune_percentage,5) 

        stage = 'reNeRF-' + str(i) 
        pre_stage = 'reNeRF-' + str(i-1)
        if start == 1: 
            pretrained = args.ckpt    
        else:
            pretrained = f'{logfolder}/{pre_stage}_finalgrid.th'

        print('pruning {} of the params'.format(prune_percentage))

        # prune + finetune
        results = run_pruning(args, val_dataset, pretrained,prune_percentage)
        model = run_finetuning(args, train_dataset, args.ckpt, results)

        # quantize   
        outpath = quantize.quantizing(model, stage, logfolder)

        # compress
        compress(outpath)
        
        # copy quantised state_dict to model before testing 
        model_clone = copy.deepcopy(model)
        model_clone.load_state_dict(torch.load(outpath))
        
        # load test data
        test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
        white_bg = test_dataset.white_bg
        ndc_ray = args.ndc_ray
        
        # test
        torch.cuda.empty_cache()
        if os.path.exists(f'{logfolder}/{stage}_test_all'):
            shutil.rmtree(f'{logfolder}/{stage}_test_all')
        
        os.makedirs(f'{logfolder}/{stage}_test_all')
        psnr,ssim, l_v = evaluation(test_dataset,model_clone, args, renderer, f'{logfolder}/{stage}_test_all',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, compute_extra_metrics= True,ndc_ray=ndc_ray,device=device)

        # save model 
        model_clone.save(f'{logfolder}/{stage}_finalgrid.th')
        
        # if performance drops too much -> stop 
        stop = False
        if psnr > best_psnr:
            best_psnr = psnr 
        elif psnr < best_psnr - 1.0:
            print("finished when prune percentage =", prune_percentage)
            stop = True

        print("=== best PSNR:", best_psnr, "===")
        if stop:
            break

