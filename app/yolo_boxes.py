def nms(boxes):
    # assuming only one image in one batch
    boxes = boxes.squeeze()
    nms_thresh = 0.45
    conf_thresh = 0.2
    no_of_valid_elems = (boxes[:, 4] > conf_thresh).nonzero().numel()
    boxes_confs_inv = 1 - boxes[:, 4]
    _, sort_ids = torch.sort(boxes_confs_inv)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    for index in range(no_of_valid_elems):
        i = sort_ids[index]
        new_ind = index + 1
        if float(boxes[i, 4]) > conf_thresh:
            xx1 = torch.max(x1[i], x1[sort_ids[new_ind:]])
            yy1 = torch.max(y1[i], y1[sort_ids[new_ind:]])
            xx2 = torch.min(x2[i], x2[sort_ids[new_ind:]])
            yy2 = torch.min(y2[i], y2[sort_ids[new_ind:]])
            w = torch.max(torch.zeros(1, device=boxes.device), xx2 - xx1 + 1)
            h = torch.max(torch.zeros(1, device=boxes.device), yy2 - yy1 + 1)
            overlap = (w * h) / area[sort_ids[new_ind:]]
            higher_nms_ind = (overlap > nms_thresh).nonzero()
            boxes[sort_ids[new_ind:][higher_nms_ind]
                  ] = torch.zeros(7, device=boxes.device)
    return boxes.unsqueeze(0)


def get_region_boxes(output):
    conf_thresh = 0.2
    num_classes = 80
    num_anchors = 5
    anchor_step = 2
    anchors_ = [1.08, 1.19,  3.42, 4.41,  6.63,
                11.38,  9.42, 5.11,  16.62, 10.52]

    anchors = torch.empty(num_anchors * 2, device=output.device)
    for i in range(num_anchors * 2):
        anchors[i] = anchors_[i]

    batch = output.size(0)
    # assert(output.size(1) == (5+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)

    output = output.view(batch*num_anchors, 5+num_classes, h*w).transpose(0,
                                                                          1).contiguous().view(5+num_classes, batch*num_anchors*h*w)

    grid_x = torch.linspace(0, w-1, w, device=output.device).repeat(h,
                                                                    1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w)
    grid_y = torch.linspace(0, h-1, h, device=output.device).repeat(w,
                                                                    1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w)
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = anchors.view(num_anchors, anchor_step).index_select(
        1, torch.zeros(1, device=output.device).long())
    anchor_h = anchors.view(num_anchors, anchor_step).index_select(
        1, torch.ones(1, device=output.device).long())
    anchor_w = anchor_w.repeat(batch, 1).repeat(
        1, 1, h*w).view(batch*num_anchors*h*w)
    anchor_h = anchor_h.repeat(batch, 1).repeat(
        1, 1, h*w).view(batch*num_anchors*h*w)
    ws = torch.exp(output[2]) * anchor_w.float()
    hs = torch.exp(output[3]) * anchor_h.float()

    det_confs = torch.sigmoid(output[4])

    cls_confs = torch.softmax(output[5: 5+num_classes].transpose(0, 1), dim=1)
    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)

    sz_hw = h * w
    boxes = torch.zeros(batch, h * w * num_anchors, 7)
    # assuming only one image in a batch
    x1 = xs / w
    y1 = ys / h
    x2 = ws / w
    y2 = hs / h
    higher_confs = ((det_confs * cls_max_confs) > conf_thresh).nonzero()
    no_selected_elems = higher_confs.numel()
    if no_selected_elems > 0:
        boxes[:, 0:no_selected_elems] = torch.stack(
            [x1, y1, x2, y2, det_confs, cls_max_confs, cls_max_ids.float()], dim=1)[higher_confs.squeeze()]
    return boxes


def boxes_from_tf(output):
    boxes = get_region_boxes(output.permute(0, 3, 1, 2).contiguous())
    boxes = nms(boxes)
    return boxes


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_area(box):
    # box = 4xn
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2):
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

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    # iou = inter / (area1 + area2 - inter)
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)


def non_max_suppression(prediction):
    conf_thres = 0.25
    iou_thres = 0.45
    classes = 80
    agnostic = False
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_det = 300  # maximum number of detections per image
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [torch.zeros((0, 6))
              ] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[
            conf.view(-1) > conf_thres]

        # Filter by class
        # if classes:
        x = x[(x[:, 5:6] == torch.tensor(classes)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5]  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]

        iou_boxes = torch_box_iou(boxes, boxes)
        iou = torch.triu(iou_boxes, diagonal=1)  # upper triangular iou matrix
        i = iou.max(0)[0] < iou_thres

        # c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # # boxes (offset by class), scores
        # boxes, scores = x[:, :4] + c, x[:, 4]
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        # if i.shape[0] > max_det:  # limit detections
        #     i = i[:max_det]
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float(
        #     ) / weights.sum(1, keepdim=True)  # merged boxes
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
    return output


def boxes_from_torch(output):
    return non_max_suppression(output)


def torch_xywh2xyxy(x):
    # Transform box coordinates from [x, y, w, h] to [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right)
    y = torch.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


# def box_area(box):
#    # box = 4xn
#    return (box[2] - box[0]) * (box[3] - box[1])


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

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
             torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(dim=2)
    # iou = inter / (area1 + area2 - inter)
    return inter.div((area1[:, None] + area2 - inter))


def torch_non_max_suppression(prediction):
    conf_thres = 0.2
    iou_thres = 0.5
    classes = None
    multi_label = False
    mode = 'vision'
    """
   Performs  Non-Maximum Suppression on inference results
   Returns detections with shape:
      nx6 (x1, y1, x2, y2, conf, cls)
   """
    # Box constraints
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096

    # nc = prediction.shape[1] - 5  # number of classes
    # multi_label &= nc > 1  # multiple labels per box
    output = []

    for xi, x in enumerate(prediction):  # image index, image inference

        # Apply conf constraint
        x = x[x[:, 4] > conf_thres]

        # Apply width-height constraint
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]

        # If none remain process next image
        if not x.shape[0]:
            output.append(torch.zeros((1, 6)))
            continue

        # Compute conf
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = torch_xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        # if multi_label:
        #  i, j = (x[:, 5:] > conf_thres).nonzero().t()
        #  x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        # else:  # best class only
        conf, j = x[:, 5:].max(1)
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
            output.append(torch.zeros((1, 6)))
            continue

        # Batched NMS
        c = x[:, 5]  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4].clone().detach() + \
            c.reshape(-1, 1) * max_wh, x[:, 4]

        iou_boxes = torch_box_iou(boxes, boxes)
        iou = torch.triu(iou_boxes, diagonal=1)  # upper triangular iou matrix
        i = iou.max(0)[0] < iou_thres

        output.append(x[i])

    return output


def boxes_from_yolo(output):
    return torch_non_max_suppression(output)
