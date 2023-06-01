import torch
import torch.nn as nn

from typing import Dict, List, Optional, Tuple

from detectron2.config import configurable, CfgNode
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling import GeneralizedRCNN
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.layers import ShapeSpec
from detectron2.data.transforms import Resize
from detectron2.layers import move_device_like

import numpy as np
import math


@META_ARCH_REGISTRY.register()
class VMRN_MR(GeneralizedRCNN):
    """
    Visual Manipulation Relationship Network
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        nn.Module.__init__(self)
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        # pixel mean [103.53, 116.28, 123.675] pixel std [1.0, 1.0, 1.0]


        # rel_layers = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),
        #     nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1),
        # )
        # in_features = 3 * 64 * 7 * 7
        # rel_fc_layers = nn.Sequential(
        #     nn.Linear(in_features, 4096),
        #     nn.ReLU(),
        #     nn.Linear(4096, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 3),
        # )
        rel_layers_o1 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.Conv2d(256, 64, 1, stride=1, padding=0),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
        )
        rel_layers_o2 = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.Conv2d(256, 64, 1, stride=1, padding=0),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
        )
        rel_layers_union = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.Conv2d(256, 64, 1, stride=1, padding=0),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
        )
        # opfc.append(self.VMRN_rel_top_union(pooled_pair[:, 2]).mean(3).mean(2))
        # torch.cat(opfc, 1)
        _input_dim = 192
        rel_fc_layers = nn.Sequential(
            nn.Linear(_input_dim, 2048),
            # nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            # nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Linear(2048, 3),
        )
        self.add_module('rel_layers_o1', rel_layers_o1)
        self.add_module('rel_layers_o2', rel_layers_o2)
        self.add_module('rel_layers_union', rel_layers_union)
        self.add_module('rel_fc_layers', rel_fc_layers)

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            0,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        # print(images.tensor.shape)
        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses, roi_feat_obj, roi_feat_union, num_obj, num_union = self.roi_heads(images, features, proposals, gt_instances)
        # _, detector_losses, nms_features, union_features, u_Boxes_idxs, proposal0_len = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        # # ## Relationship Detection
        ####   CHECK_BEFORE_TRAIN   ####
        # roi_feat_obj = roi_feat_obj.detach()
        # roi_feat_union = roi_feat_union.detach()

        rel_loss = torch.tensor(0.).float().cuda()
        rel_criterion = nn.CrossEntropyLoss()   # num_classes = 3   # 0: Parent / 1: Child / 2: None

        union_idx = 0
        for B in range(len(batched_inputs)):
            rel_mat = batched_inputs[B]['rel_mat']
            pred_rel_mat = torch.zeros_like(rel_mat)
            if B == 0:
                roi_feat_batch_obj = roi_feat_obj[:num_obj]
            else:
                roi_feat_batch_obj = roi_feat_obj[num_obj:]
            if len(roi_feat_batch_obj) <= 1:
                continue
            obj_pair_feat = []
            gts = []
            for i, feat_i in enumerate(roi_feat_batch_obj):
                for j, feat_j in enumerate(roi_feat_batch_obj):
                    if i == j: continue
                    
                    opfc = []
                    opfc.append(self.rel_layers_o1(feat_i.unsqueeze(0)).mean(3).mean(2))
                    opfc.append(self.rel_layers_o2(feat_j.unsqueeze(0)).mean(3).mean(2))
                    opfc.append(self.rel_layers_union(roi_feat_union[union_idx].unsqueeze(0)).mean(3).mean(2))
                    
                    union_idx += 1

                    obj_pair_feat.append(torch.cat(opfc, 1))
                    gt = self.rel_gt(rel_mat[i][j], rel_mat[j][i])
                    gts.append(gt)

            obj_pair_feat = torch.cat(obj_pair_feat, 0)
            output = self.rel_fc_layers(obj_pair_feat)
            pred = torch.argmax(output, dim=1)
            pred_idx = 0
            for i, feat_i in enumerate(roi_feat_batch_obj):
                for j, feat_j in enumerate(roi_feat_batch_obj):
                    if i == j: continue
                    if pred[pred_idx] == 0:     # i is parent of j
                        pred_rel_mat[i][j] = -1
                    elif pred[pred_idx] == 1:   # i is child of j
                        pred_rel_mat[i][j] = 1
            print('@@ Pred Rel Mat \t\t vs \t\t GT Rel Mat')
            gts_mask = []
            for i in range(len(pred_rel_mat)):
                print(pred_rel_mat[i], '\t\t\t', rel_mat[i])
                gts_mask.append(gts[i] < 2)
            print()
            ####   CHECK_BEFORE_TRAIN   ####
            rel_loss += rel_criterion(output, torch.tensor(gts).view(-1).cuda())
            # REL_LOSS_W = 9
            # for i, (o, g) in enumerate(zip(output, gts)):
            #     if g < 2:
            #         rel_loss += REL_LOSS_W*rel_criterion(o.unsqueeze(0), torch.tensor(g).view(-1).cuda())
            #     else:
            #         rel_loss += rel_criterion(o.unsqueeze(0), torch.tensor(g).view(-1).cuda())



        
        ####   CHECK_BEFORE_TRAIN   ####
        if len(roi_feat_union):
            rel_loss /= len(roi_feat_union)
        

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update({'loss_rel':rel_loss})      ####   CHECK_BEFORE_TRAIN   ####
        return losses


    def rel_gt(self, AB, BA):
        # AB: A=row, B=column
        # return A's class
        if AB == -1: return 0   # Parent
        elif BA == -1: return 1 # Child
        return 2   # None
        

    def inference(
            self,
            batched_inputs: List[Dict[str, torch.Tensor]],
            detected_instances: Optional[List[Instances]] = None,
            do_postprocess: bool = True,
        ):
            """
            Run inference on the given inputs.

            Args:
                batched_inputs (list[dict]): same as in :meth:`forward`
                detected_instances (None or list[Instances]): if not None, it
                    contains an `Instances` object per image. The `Instances`
                    object contains "pred_boxes" and "pred_classes" which are
                    known boxes in the image.
                    The inference will then skip the detection of bounding boxes,
                    and only predict other per-ROI outputs.
                do_postprocess (bool): whether to apply post-processing on the outputs.

            Returns:
                When do_postprocess=True, same as in :meth:`forward`.
                Otherwise, a list[Instances] containing raw network outputs.
            """
            assert not self.training

            images = self.preprocess_image(batched_inputs)
            if "annotations" in batched_inputs[0]:
                gt_instances = []

                ####   CHECK_BEFORE_TRAIN   ####
                image_size = (800, 800)
                # image_size = (640, 480)
                for x in batched_inputs:
                    gt_boxes = []
                    for y in x['annotations']:
                        gt_bbox = torch.tensor(y['bbox']).unsqueeze(0).cuda()
                        if gt_boxes == []:
                            gt_boxes = gt_bbox
                        else:
                            torch.cat([gt_boxes, gt_bbox], dim=0)
                        # print(gt_bbox)
                        # print(gt_bboxes)
                    gt_instance = Instances(image_size)
                    gt_instance.set('gt_boxes', Boxes(gt_boxes))
                    gt_instances.append(gt_instance)
            else:
                gt_instances = None

            features = self.backbone(images.tensor)

            if detected_instances is None:
                if self.proposal_generator is not None:
                    proposals, _ = self.proposal_generator(images, features, None)
                else:
                    assert "proposals" in batched_inputs[0]
                    proposals = [x["proposals"].to(self.device) for x in batched_inputs]

                results, roi_feat_obj, roi_feat_union, num_obj, num_union = self.roi_heads(images, features, proposals, gt_instances)

                ## Relationship Detection

                union_idx = 0
                # acc, acc_total = 0, 0
                for B in range(len(batched_inputs)):
                    # rel_mat = batched_inputs[B]['rel_mat']
                    pred_rel_mat = torch.zeros(len(batched_inputs[B]['rel_mat']), len(batched_inputs[B]['rel_mat']))
                    diff, total = 0, 0
                    if B == 0:
                        roi_feat_batch_obj = roi_feat_obj[:num_obj]
                    else:
                        roi_feat_batch_obj = roi_feat_obj[num_obj:]
                    if len(roi_feat_batch_obj) <= 1:
                        continue
                    for i, feat_i in enumerate(roi_feat_batch_obj):
                        for j, feat_j in enumerate(roi_feat_batch_obj):
                            if i == j: continue
                            
                            output_i = self.rel_layers_o1(feat_i.unsqueeze(0))
                            output_j = self.rel_layers_o2(feat_j.unsqueeze(0))
                            output_u = self.rel_layers_union(roi_feat_union[union_idx].unsqueeze(0))
                            union_idx += 1

                        flatten_i = torch.flatten(output_i)
                        flatten_j = torch.flatten(output_j)
                        flatten_u = torch.flatten(output_u)
                        flatten = torch.cat((flatten_i, flatten_j, flatten_u)).unsqueeze(0)
                        output = self.rel_fc_layers(flatten)
                        predict = torch.argmax(output, dim=1)
                        if predict == 0:        # parent
                            pred_rel_mat[i][j] = -1
                        elif predict == 1:      # child
                            pred_rel_mat[j][i] = -1
                        elif predict == 2:
                            pass
                        # gt = self.rel_gt(rel_mat[i][j], rel_mat[j][i])
                        
                    #     diff += math.pow(output - gt, 2)
                    #     total += 1
                    
                    # if total:
                    #     acc += (diff / total)
                    #     acc_total += 1

            else:
                detected_instances = [x.to(self.device) for x in detected_instances]
                results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

            if do_postprocess:
                assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
                return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes), pred_rel_mat
            return results