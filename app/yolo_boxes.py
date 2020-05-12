def torch_xywh2xyxy(x):
    # Transform box coordinates from [x, y, w, h] to [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right)
    y = torch.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_area(box):
    # box = 4xn
    return (box[2] - box[0]) * (box[3] - box[1])


def torch_box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    area1 = box_area(box1.transpose(1, 0))
    area2 = box_area(box2.transpose(1, 0))

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(dim=2)
    return inter.div((area1[:, None] + area2 - inter))  # iou = inter / (area1 + area2 - inter)


def torch_non_max_suppression(prediction):
    conf_thres = 0.1
    iou_thres = 0.6
    classes = None
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """
    # Box constraints
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height

    nc = prediction.shape[1] - 5  # number of classes
    output = []
    for xi, x in enumerate(prediction):  # image index, image inference

        # Apply conf constraint
        x = x[x[:, 4] > conf_thres]

        # Apply width-height constraint
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = torch_xywh2xyxy(x[:, :4])
        # Detections matrix nx6 (xyxy, conf, cls)
        conf = x[:, 5:].max(1).values
        j = x[:, 5:].max(1).indices
        x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)

        # Filter by class
        if classes is not None:
            x = x[(j.reshape(-1, 1) == classes).any(1)]

        # Apply finite constraint
        if not torch.isfinite(x).all():
            x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5]  # classes
        boxes, scores = x[:, :4].clone().detach() + c.reshape(-1, 1) * max_wh, x[:,
                                                                               4]  # boxes (offset by class), scores
        iou_boxes = torch_box_iou(boxes, boxes)
        iou = torch.triu(iou_boxes, diagonal=1)  # upper triangular iou matrix
        i = iou.max(0, keepdim=True)[0] < iou_thres

        output.append(x[i.squeeze()])

    return output


def boxes_from_yolo(output):
    return torch_non_max_suppression(output)
