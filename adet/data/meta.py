import io
import json
import logging
import contextlib
import numpy as np


from PIL import Image
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

from detectron2.structures import BoxMode, PolygonMasks, Boxes
from detectron2.data import MetadataCatalog, DatasetCatalog
from pycocotools import mask as maskUtils


logger = logging.getLogger(__name__)

__all__ = ["load_meta_json"]

def load_segm(anno, type):
    segm = anno.get(type, None)
    if isinstance(segm, dict):
        if isinstance(segm["counts"], list):
            # convert to compressed RLE
            segm = maskUtils.frPyObjects(segm, *segm["size"])
    else:
        # filter out invalid polygons (< 3 points)
        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
        if len(segm) == 0:
            num_instances_without_valid_segmentation += 1
            segm = None
    return segm

def load_meta_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    ### adet/data/uoais.py/load_uoais_json
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    
    #### TODO: implement this function ####
    id_map = None

    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []
    ann_keys = ["bbox", "category_id", "visible_bbox", "area"] + (extra_annotation_keys or [])

    num_instances_without_valid_segmentation = 0
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = image_root + img_dict["file_name"]
        record["depth_file_name"] = image_root + img_dict["depth_file_name"]
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["rel_mat"] = img_dict["rel_mat"]
        image_id = record["image_id"] = img_dict["image_id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if anno.get("segmentation", None):  # either list[list[float]] or dict(RLE)
                obj["segmentation"] = anno.get("segmentation", None)
            if anno.get("visible_mask", None): 
                obj["visible_mask"] = anno.get("visible_mask", None)
            if anno.get("occluded_mask", None):
                obj["occluded_mask"] = anno.get("occluded_mask", None)
            obj["occluded_rate"] = anno.get("occluded_rate", None)

            obj["bbox_mode"] = BoxMode.XYWH_ABS     # (x0, y0, w, h)
        
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
            "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts



def visualize_rel_mat(rel_mat, objs):
    classes = ['__background__',  # always index 0
               'sugar_box', 'tomato_soup_can', 'mustard_bottle', 'potted_meat_can', 'banana', 
               'bowl', 'mug', 'power_drill', 'scissor', 'chips_can', 'strawberry', 'apple', 
               'lemon', 'peach', 'pear', 'orange', 'plum', 'knife', 'phillips_screwdriver', 
               'flat_screwdriver', 'racquetball', 'b_cups', 'd_cups', 'a_toy_airplane', 
               'c_toy_airplane', 'd_toy_airplane', 'f_toy_airplane', 'h_toy_airplane', 'i_toy_airplane', 
               'j_toy_airplane', 'k_toy_airplane', 'light_bulb', 'no_class', 'kitchen_knife', 
               'screw_valve', 'plastic_pipes', 'cables_in_transparent_bag', 'cables', 'wire_cutter',
               'desinfection', 'hairspray', 'handcream', 'toothpaste', 'toydog', 'sponge', 
               'pneumatic_cylinder', 'airfilter', 'coffeefilter', 'wash_glove', 'wash_sponge', 
               'garbage_bags', 'deo', 'cat_milk', 'bottle_glass', 'bottle_press_head', 'shaving_cream',
               'chewing_gum_with_spray', 'lighters', 'cream_soap', 'box_1', 'box_2', 'box_3', 'box_4', 
               'box_5', 'box_6', 'box_7', 'box_8', 'glass_cup', 'tennis_ball', 'cup', 'wineglass', 
               'handsaw', 'lipcare', 'woodcube_a', 'lipstick', 'nosespray', 'tape', 'bookholder', 
               'clamp', 'glue', 'stapler', 'calculator', 'clamp_small', 'clamp_big', 'glasses', 
               'crayons', 'marker_big', 'marker_small', 'greek_busts', 'object_wrapped_in_foil', 
               'water_bottle_deformed', 'bubble_wrap', 'woodblock_a', 'woodblock_b', 'woodblock_c', 
               'mannequin', 'cracker_box']
    
    categs = [classes[obj] for obj in objs]
    print(objs)
    print(categs)
    for _ in rel_mat:
        print(_)
    print()


if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger
    from detectron2.utils.visualizer import Visualizer
    import os, sys
    import tqdm
    

    logger = setup_logger(name=__name__)
    # print(DatasetCatalog.list())

    dicts = load_meta_json("/ailab_mat/dataset/MetaGraspNet/Annotations/meta_sim_train.json",
                           "datasets/MetaGraspNet/dataset_sim/")
    logger.info("Done loading {} samples.".format(len(dicts)))
    print("Done loading {} samples.".format(len(dicts)))

    dirname = "meta-data-vis"
    os.makedirs(dirname, exist_ok=True)
    i = 0
    for d in (dicts):
        print(d["file_name"])
        img = Image.open(d["file_name"])
        visualizer = Visualizer(img)
        vis = visualizer.draw_dataset_dict(d)
        fpath = os.path.join(dirname, os.path.basename(d["file_name"]))
        vis.save(fpath)

        objs = [obj["category_id"] for obj in d["annotations"]]
        visualize_rel_mat(d["rel_mat"], objs)
        i+=1
        if i==5: break