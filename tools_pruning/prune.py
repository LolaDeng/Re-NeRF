import torch
import torch.nn.utils.prune as prune

def pruning(model,target_layers,prune_percentage,add_neighbours = True):
    
    no_layers =3 
    importance_maps_merge = dict()
    
    for i in range(no_layers):
        parameters_to_prune = []
        importance_maps = dict()
        for module_name, module in model.named_modules():
            if module_name in target_layers:
                parameters_to_prune.append((module, str(i)))
                taylor= torch.abs(module[i].grad * module[i])
                importance_maps[module_name] =  taylor /torch.max(taylor)

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=prune_percentage,
            importance_scores= importance_maps
            ) 
        importance_maps_merge.update(importance_maps)

    results = dict()
    for key in importance_maps_merge.keys():
        mask_name = key + '_mask'
        if add_neighbours:
            output_neighbours= neighbours(model, target_layers, device = "cuda")
            results.update(output_neighbours)
        else:
            results[key] = model.state_dict()[mask_name]    

    return results

def neighbours(model, target_layers, device):

    no_layers =3 
    results = dict()
    has_neighbours = torch.nn.AvgPool3d(3,stride=1,padding=1)
    for i in range(no_layers):
        for layer in target_layers:
            if layer == "density_line" or layer == "app_line":
                continue 
            ori_key = ".".join([layer, str(i) + '_orig']) #density_plane.0_orig
            mask_key = ".".join([layer, str(i) + '_mask']) #density_plane.0_mask
            mask_name = ".".join([layer, str(i)])
            if mask_key in model.state_dict().keys():
                mask = model.state_dict()[mask_key]
                for param_name, param in model.named_parameters(): 
                    if param_name == ori_key:
                        param_ori = param 
                # update neighbours 
                abs_imp= torch.abs(param_ori.grad)
                adding = True
                while adding:
                    avg_imp = torch.sum(abs_imp * mask) / torch.sum(mask) 
                    output = has_neighbours(mask)
                    delta = 1.0
                    add_neighbours = ((output * inverse_mask(mask) * abs_imp)) * delta > avg_imp
                    new_mask = mask.logical_or(add_neighbours) 
                    num_added = new_mask.sum() - mask.sum() # log the number of neighbours added 

                    if num_added > 0 :
                        mask = new_mask.bool().to(torch.float32).clone()
                        mask = mask.to(device)
                    else:
                        adding = False

                results[mask_name] = new_mask
    return results

def inverse_mask(mask): 
    res = mask.clone()
    res[mask==0] = 1 
    res[mask!=0] = 0
    return res 


             





