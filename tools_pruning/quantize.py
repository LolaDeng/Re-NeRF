import torch
import os

def quantizing(model, stage, basedir): 
   
    fname = stage + "_state_dict.tar"
    state_dict_path = os.path.join(basedir, fname)
    torch.save(model.state_dict(),state_dict_path)

    st_dict = torch.load(state_dict_path, map_location="cpu")
    quantizing_st_dict = {}
    for key in st_dict:
        if not "grid" in key:
            quantizing_st_dict[key] = st_dict[key]
        else:
            non_zeros = (st_dict[key] != 0)*1.0
            quantizing_st_dict[key] = torch.quantize_per_tensor(st_dict[key], 1, 0, torch.qint8)
            st_dict[key] = ((torch.dequantize(quantizing_st_dict[key])) * non_zeros)

    outfname = stage + "_quantized_state_dict.tar"
    outpath = os.path.join(basedir, outfname)
    torch.save(st_dict,outpath)
    print(f'Quantizing ({stage}): saved state_dict at', outpath)

    return outpath
