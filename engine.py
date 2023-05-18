
 
"""
Train and eval functions used in main.py
"""
import math
import os
from re import I, S
import sys
import time
from typing import Iterable
 
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.open_world_eval import OWEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, generalized_box_iou
from util.plot_utils import plot_prediction, plot_prediction_GT, res2xml
import matplotlib.pyplot as plt
from copy import deepcopy



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, nc_epoch: int, max_norm: float = 0):
    # Setting in train mode
    model.train()
    criterion.train() #criterion most likely a loss function
    # Set up the logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    # Fetch data
    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()
    last_loss = 0

    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        if epoch >= criterion.nc_epoch:
            Iter = (epoch-criterion.nc_epoch) * len(data_loader) + _
        else:
            Iter = 0
        loss_dict = criterion(samples, outputs, targets, epoch, Iter, last_loss) ## samples variable needed for feature selection

        weight_dict = deepcopy(criterion.weight_dict)
        ## condition for starting nc loss computation after certain epoch so that the F_cls branch has the time
        ## to learn the within classes seperation.

        if epoch < nc_epoch: 
            for k,v in weight_dict.items():
                if 'NC' in k:
                    weight_dict[k] = 0
                if 'cluster' in k:
                    weight_dict[k] = 0
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        last_loss = losses.item()

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        ## Just printing NOt affectin gin loss function
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
 
        loss_value = losses_reduced_scaled.item()
 
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
 
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
 
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
 
        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

## ORIGINAL FUNCTION
@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, args):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = OWEvaluator(base_ds, iou_types, args=args)
 
    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
 
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
 
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)
 
        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name
 
            panoptic_evaluator.update(res_pano)
 
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()
     # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        metrics = coco_evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}


    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]


    return stats, coco_evaluator, metrics
 
@torch.no_grad()
def viz(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    criterion.eval()
 
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
 
    for samples, targets in data_loader:
        #print(f"Targets: {targets}")
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #print(targets)
        top_k = len(targets[0]['boxes'])
        #top_k = 2
        print(f"Applying CAT to: { int(targets[0]['image_id'][0])}.jpg")
        
        #print(f"Target instances:{top_k}")
        #print("Apply CAT on image...")
        outputs = model(samples)
        
        # indices, logits and predicted boxes will not be used in the plot function
        indices = outputs['pred_logits'][0].softmax(-1)[..., 1].sort(descending=True)[1][:top_k]
        predictied_boxes = torch.stack([outputs['pred_boxes'][0][i] for i in indices]).unsqueeze(0)
        logits = torch.stack([outputs['pred_logits'][0][i] for i in indices]).unsqueeze(0)
        fig, ax = plt.subplots(1, 3, figsize=(10,3), dpi=200)
 
        img = samples.tensors[0].cpu().permute(1,2,0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = (img * 255)
        img = img.astype('uint8')
        h, w = img.shape[:-1]

        # Postprocess: custom written
        results = postprocessors['bbox'](outputs, torch.tensor([[h,w]]).to(device))

        # Slice the results to only show the first top_k predictions (CUSTOM)
        scores = results[0]['scores'][0:top_k]
        labels = results[0]['labels'][0:top_k]
        boxes = results[0]['boxes'][0:top_k]


        # Pred results
        plot_prediction(samples.tensors[0:1],boxes=boxes, scores = scores,labels=labels ,ax=ax[1], plot_prob=False)
        ax[1].set_title('Prediction CAT')
 
        #print(f"Bounding boxes of cat {boxes}")
        #print(f"Bounding boxes of targets: {targets[0]['boxes']}")
        #print(f"Labels of targets: {targets[0]['labels']}")
        # GT Results
        #print(f"Target boxes: {targets[0]['boxes']}")
        #plot_prediction_GT(samples.tensors[0:1], boxes=targets[0]['boxes'], labels=targets[0]['labels'], ax=ax[2], plot_prob=False)
        #plot_prediction(samples.tensors[0:1], boxes=targets[0]['boxes'], scores=scores,labels=targets[0]['labels'], ax=ax[2], plot_prob=False)
        #ax[2].set_title('GT')
 
        for i in range(3):
            ax[i].set_aspect('equal')
            ax[i].set_axis_off()
        print(f'Saving image:{int(targets[0]["image_id"][0])}')
        plt.savefig(os.path.join(output_dir, f'img_{int(targets[0]["image_id"][0])}.jpg'))
        
        # Store the results in an xml format.
        res2xml(scores, labels, boxes, output_dir, f"{targets[0]['image_id'][0]}.jpg", h,w)

        print(len(boxes))
        print(torch.where(labels==80))
        u_instances = torch.where(labels==80)[0]

        path_unknown = os.path.join(output_dir,'unknown', str(int(targets[0]["image_id"][0])))
        
        for i, box in enumerate(boxes): 
            if i in u_instances:
                print(f"Unknown box {i}: {box}")
                box = torch.abs(torch.Tensor.int(box))
                plt.figure(2)
                plt.axis('off')
                plt.imshow(img[box[1]:box[3],box[0]:box[2]])
                
                if os.path.exists(path_unknown)==False:
                    os.mkdir(path_unknown)
                plt.savefig(os.path.join(path_unknown, f'unknown_{int(targets[0]["image_id"][0])}_{i}.jpg'),bbox_inches='tight')
            
        #unknown_boxes = boxes[torch.where(labels == 80)]

        # Filter same proposals
        print(f"Sorting boxes out ...")
        iou = generalized_box_iou(boxes, boxes)

        # Find instances for overlapping windows
        coord = torch.where(iou > 0.6) 
        # Filter the selfcoresponding images out.
        index = torch.where(coord[0]!=coord[1]) 
        x = coord[0][index[0]]
        y = coord[1][index[0]]
        
        contains_x = torch.any(torch.eq(x.unsqueeze(1), u_instances), dim=1)
        contains_y = torch.any(torch.eq(y.unsqueeze(1), u_instances), dim=1)
        #contains_x = torch.any(torch.eq( u_instances.unsqueeze(1), x.unsqueeze(0)), dim=1)
        #contains_y = torch.any(torch.eq( u_instances.unsqueeze(1), y.unsqueeze(0)), dim=1)
        print(x)
        print(y)

        print(contains_x)
        print(contains_y)
        print(torch.where(contains_x))
        x_unknown = x[torch.where(contains_x)]
        y_unknown = y[torch.where(contains_y)]
        
        #unknown_scores = scores[torch.where(labels == 80)]
        # Keep the results of the proposals with higher scores
        #destroy = x[torch.where(scores[x]<scores[y])]
        destroy = x_unknown[torch.where(scores[x_unknown]<scores[y_unknown])]

        # Delete cropped pictures 
        for i in torch.unique(destroy):
            if i in u_instances:
                print(f'Deleting file: unknown_{int(targets[0]["image_id"][0])}_{i}.jpg')
                os.remove(os.path.join(path_unknown, f'unknown_{int(targets[0]["image_id"][0])}_{i}.jpg'))
            
        


# Define a recursive function to return the shape of every tensor in a nested structure
def get_nested_shapes(nested_tensor):
    if isinstance(nested_tensor, torch.Tensor):
        return [nested_tensor.size()]
    else:
        shapes = []
        for t in nested_tensor:
            shapes.extend(get_nested_shapes(t))
        return shapes