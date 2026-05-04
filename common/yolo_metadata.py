def select_detection_for_object(result, obj_id):
    target_cls = obj_id - 1
    best_box, best_conf = None, -1.0
    for box in result.boxes:
        if int(box.cls) != target_cls:
            continue
        conf = float(box.conf)
        if conf > best_conf:
            best_box, best_conf = box, conf
    return best_box


def get_object_metadata(models_info, obj_id):
    return models_info.get(obj_id) or models_info.get(str(obj_id))
