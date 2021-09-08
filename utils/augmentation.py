
import numpy as np
import tensorflow as tf
import cv2
from utils.box_utils import compute_iou, compute_area
import logging, time

def horizontal_flip(image, boxes, labels):
    """ Function to horizontally flip the image
        The gt boxes will be need to be modified accordingly

    Args:
        image: the original PIL Image
        boxes: gt boxes tensor (num_boxes, 4)
        labels: gt labels tensor (num_boxes,)

    Returns:
        image: the horizontally flipped PIL Image
        boxes: horizontally flipped gt boxes tensor (num_boxes, 4)
        labels: gt labels tensor (num_boxes,)
    """
    image = image[:,::-1]
    boxes = np.stack([1-boxes[:, 2], boxes[:,1], 1-boxes[:, 0], boxes[:, 3]], axis=1)

    return image, boxes, labels

def rotate_image(image, angle, border_color=None):
    # grab the dimensions of the image and then determine the
    # center
    if border_color == None:
        border_color=(0, 0, 0)

    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=border_color), M

def warp_box(box, M, image_size):
    x_min, y_min, x_max, y_max = box
    h, w = image_size
    x1, y1 = np.matmul((x_min, y_min), M[:,:2].T)
    x2, y2 = np.matmul((x_max, y_max), M[:,:2].T)
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    return x_min, y_min, x_max, y_max

def get_corners(box):
    x_min, y_min, x_max, y_max = box
    return x_min, y_min, x_max, y_min, x_min, y_max, x_max, y_max

def warp_corners(corner, M):
    x1,y1,x2,y2,x3,y3,x4,y4 = corner
    x1, y1 = np.matmul((x1, y1), M[:,:2].T)
    x2, y2 = np.matmul((x2, y2), M[:,:2].T)
    x3, y3 = np.matmul((x3, y3), M[:,:2].T)
    x4, y4 = np.matmul((x4, y4), M[:,:2].T)
    return x1,y1,x2,y2,x3,y3,x4,y4

def corner2normbbox(corner, size):
    x1,y1,x2,y2,x3,y3,x4,y4 = corner
    h,w = size
    x_min, x_max = min(x1, x2, x3, x4), max(x1, x2, x3, x4)
    y_min, y_max = min(y1, y2, y3, y4), max(y1, y2, y3, y4)
    return np.clip((x_min/w, y_min/h, x_max/w, y_max/h),0,1)
    #return x_min/w, y_min/h, x_max/w, y_max/h
def rotate(image, angle, boxes):
    R, M = rotate_image(image.copy(), angle)
    h,w = image.shape[:2]
    H,W = R.shape[:2]
    rotated_image = R[(H//2)-(h//2):(H//2)+(h//2), (W//2)-(w//2):(W//2)+(w//2)]
    denormalized_boxes = boxes*np.array([w,h,w,h])
    corners = list(map(lambda box : get_corners(box), denormalized_boxes))
    corners = np.array(corners)
    centered_corners = corners - np.array([w//2, h//2, w//2, h//2, w//2, h//2, w//2, h//2])
    rotated_centered_corners = list(map(lambda box : warp_corners(box, M), centered_corners))
    rotated_corners = rotated_centered_corners + np.array([w//2, h//2, w//2, h//2, w//2, h//2, w//2, h//2])
    rotated_boxes = list(map(lambda c : corner2normbbox(c, (h,w)), rotated_corners))
    rotated_boxes = np.array(rotated_boxes)
    return rotated_image, rotated_boxes


def random_patching(img, boxes, labels, thresh = 0.6):
    count = 0
    success = False
    while True:
        count+=1
        patch_w = np.random.uniform(95/110, 105/110)
        patch_h = np.random.uniform(90/120, 110/120)
        patch_xmin = np.random.uniform(0, 1 - patch_w)
        patch_ymin = np.random.uniform(0, 1 - patch_h)
        patch_xmax = patch_xmin + patch_w
        patch_ymax = patch_ymin + patch_h
        patch = np.array([[patch_xmin, patch_ymin, patch_xmax, patch_ymax]],
                dtype=np.float32)
        patch = np.clip(patch, 0.0, 1.0)
        top_left = tf.math.maximum(boxes[:, :2], patch[:, :2])
        bot_right = tf.math.minimum(boxes[:, 2:], patch[:, 2:])
        overlap_area = compute_area(top_left, bot_right)
        box_area = compute_area(boxes[:, :2], boxes[:, 2:])
        if count > 99:
            return img, boxes, labels, success
        if tf.math.reduce_any((overlap_area/box_area) >= thresh):
            included_boxes = ((overlap_area/box_area) >= thresh)
            boxes = tf.boolean_mask(boxes, included_boxes)
            labels = tf.boolean_mask(labels, included_boxes)
            patch = patch[0]
            x1, y1 = (int(patch[0]*img.shape[1]), int(patch[1]*img.shape[0])) # (x1, y1)
            x2, y2 = (int(patch[2]*img.shape[1]), int(patch[3]*img.shape[0])) #(x2, y2)
            boxes = tf.stack([
                (boxes[:, 0] - patch[0]) / patch_w,
                (boxes[:, 1] - patch[1]) / patch_h,
                (boxes[:, 2] - patch[0]) / patch_w,
                (boxes[:, 3] - patch[1]) / patch_h], axis=1)
            boxes = tf.clip_by_value(boxes, 0.0, 1.0)
            #print(patch_w, patch_h)
            success=True
            return img[y1:y2, x1:x2], boxes, labels, success

def random_image_color(img):
    #img = img[tf.newaxis,:,:,:]
    #ref : https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/#color
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, 0.5)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    return np.array(img)

def augment(image, boxes, labels, random_crop_rate=0.6):
    flip = np.random.choice([0, 1])
    if flip:
        image, boxes, labels = horizontal_flip(image, boxes, labels)

    rotation_angle = np.random.uniform(-15,15)
    rot_image, rot_boxes = rotate(image, rotation_angle, boxes)
    image, boxes, labels, success = random_patching(rot_image, rot_boxes, labels, random_crop_rate)
    if success == False:
        logging.basicConfig(filename='example.log', level=logging.DEBUG)
        logging.warning('patching fails :'+str(time.localtime()))
        logging.warning(boxes)
        logging.warning(labels)
        print(success)
    color = np.random.choice([0, 1])
    if color:
        image = random_image_color(image)
    return image, boxes, labels
