import numpy as np
from utils.parse_label_map import create_category_index_dict
from utils.visualization_utils import prepare_visualization

def inference(input_image, device="x86", width=1200, height=1200, thresh=0.5):
    label_map_file = "utils/mscoco_label_map.pbtxt"
    label_map = create_category_index_dict(label_map_file)
    if device == "x86":
        from detectors.x86_detector import X86Detector
        detector = X86Detector(width=width, height=height, thresh=thresh)
    else:
        raise ValueError("device should be one of 'x86', 'edgetpu' or 'jetson' but you provided {0}".format(device))
    
    model_path = None
    detector.load_model(model_path, label_map)
    image = np.array(input_image)
    preprocessed_image = detector.preprocess(image)
    result = detector.inference(preprocessed_image)
    output_dict = prepare_visualization(result)

    out = []
    item={}

    for obj in result.objects:
        item["category"] = obj.category
        item["bbox"] = [obj.bbox.left, obj.bbox.top, obj.bbox.right, obj.bbox.bottom]
        if obj.bbox.HasField("score"):
            item["score"] = obj.bbox.score
        else:
            item["score"] = 1.0
        
        out.append(item)

    return out
