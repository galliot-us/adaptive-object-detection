from protos.label_map_pb2 import StringIntLabelMap
from google.protobuf import text_format
import os


def parse_label_map(pb_txt_file):
    label_map = StringIntLabelMap()
    if not os.path.isfile(pb_txt_file):
        raise FileNotFoundError("the provided label map {} is not exist".format(pb_txt_file))
    with open(pb_txt_file, "rb") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, label_map)
    return label_map


def create_category_index_dict(pb_txt_file):
    label_map = parse_label_map(pb_txt_file)
    classes_map = {item.id: {"id": item.id, "name": item.name} for item in label_map.item}
    return classes_map
