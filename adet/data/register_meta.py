from detectron2.data import DatasetCatalog, MetadataCatalog

from .meta import load_meta_json

__all__ = ["register_meta_instances"]

def register_meta_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in MetaGraspNet format for instance detection.
    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2017_train".
        metadata (dict): extra metadata associated with this dataset. Must contain
            "thing_classes" field.
        json_file (str): path to the json instance annotation file.
        image_root (str): the directory where the images in the dataset are stored.
    """
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_meta_json(json_file, image_root))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="vmrn", **metadata
    )