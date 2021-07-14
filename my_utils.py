import cv2 # version 4.2.0
import os
import torch # Pytorch version 1.3.1
import glob
import numpy as np
from datetime import datetime

"""
File name: my_utils.py
Author: Tunai P. Marques
Website: tunaimarques.com | github.com/tunai
Date last modified: July 13 2021
Descrition: bundle of helper functions for Machine Learning/Deep Learning

    SUMMARY OF FUNCTIONS:

    print_gpu_stats: reports on GPU memory usage. Use it in the beginning of each epoch of a training routine for improved
    memory monitoring.
    iou_bbs: returns the IoU between two bounding boxes (third party by Adrian Rosebrock, 2016).
    iou_segm: returns the pixel-level intersection-over-union (IoU) between a prediction and a ground truth mask.
    mse: returns the mean squared error (mse) between two images.
    show_img: generic printing function for images of different formats using OpenCV.
    filter_detectron2_detections: filter detectron2's detections.
    concatenate_bbs: concatenates outputs composed by multiple bounding boxes.
    plot_all_bb: plot all the detection boundng boxes found for a given image.
    read_all_images: read all image files from a folder.
    check_create_dir: check if a directory exists and creates it if it doesn't.
    create_metadata: creates a metadata file summarizing the parameters and performance metrics of a DL model.

"""

def print_gpu_stats(hardware_id=0):
    '''
    Prints memory and identification information about the device on hardware_id (dafault 0)

    :param hardware_id: int specifying the hardware being considered (default 0)
    :return: N/A
    '''

    name = torch.cuda.get_device_name()
    total_memory = torch.cuda.get_device_properties(hardware_id).total_memory / 1e9
    reserved_memory = torch.cuda.memory_cached(hardware_id) / 1e9
    allocated_memory = torch.cuda.memory_allocated(hardware_id) / 1e9
    free_memory = reserved_memory - allocated_memory
    print('\nGPU memory stats for {}:'.format(name))
    print('total:{:.2f}gb | reserved:{:.2f}gb | allocated:{:.2f}gb | free in reserved:{:.2f}gb'.format(total_memory,
                                                                                                       reserved_memory,
                                                                                                       allocated_memory,
                                                                                                       free_memory))


def iou_bbs(boxA, boxB):  # third-party
    '''
    Calculates the intersection over union of two bounding boxes (third party)

    Third-party code by Adrian Rosebrock, 2016.
    Available at: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    Accessed on May 28, 2020

    :param boxA: first bounding box — a list of four numbers: (x,y) coordinates of top-let and bottom right coordinates
    :param boxB: second bounding box — a list of four numbers: (x,y) coordinates of top-let and bottom right coordinates
    :return: IoU between boxA and boxB
    '''

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def iou_segm(prediction, gt_mask, thresh=0.4, verbose=False, binarize=True):
    '''
    Determines the pixel-level IoU between a prediction and a ground-truth mask

    :param prediction: prediction mask (torch tensor or numpy array)
    :param gt_mask: ground truth mask (numpy array)
    :param thresh: threshold above which predictions are considered to be from the positive class. All other pixels are
    turn into 0 considering binarize=True
    :param verbose: flag for the printing of detailed information
    :param binarize: flag for converting the prediction into a binary image
    :return: pixel-level IoU between prediction and gt_mask
    '''

    if torch.is_tensor(prediction['out']):  # if this is a tensor, transfer it to the cpu and convert it to a np array
        prediction = prediction['out'].cpu().detach().numpy()[0][0]

    if binarize:
        prediction[prediction >= thresh] = 255
        prediction[prediction != 255] = 0

    intersection = np.logical_and(gt_mask.astype(bool), prediction.astype(bool))
    union = np.logical_or(gt_mask.astype(bool), prediction.astype(bool))

    if verbose:
        print('Non-zero pixels in the prediction: {}'.format(cv2.countNonZero(prediction)))
        print('Non-zero pixels in the mask: {}'.format(cv2.countNonZero(gt_mask)))
        print('Number of intersections (pixels): {}'.format(np.count_nonzero(intersection)))
        print('Number of unions (pixels): {}'.format(np.count_nonzero(union)))

    return np.sum(intersection) / np.sum(union)


def mse(imageA, imageB):
    '''
    Calculates the Mean Square Error between two images
    (for SSIM use skimage.metrics - structural_similarity)

    :param imageA: first image (numpy array)
    :param imageB: second image (numpy array)
    :return: MSE between imageA and imageB
    '''

    # the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
    # NOTE: the two images must have the same spatial dimensions

    assert (imageA.shape == imageB.shape), 'The images have different shapes: imageA - {}, imageB - {}'.format(
        imageA.shape, imageB.shape)
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE: the lower the error, the more "similar" the two images are
    return err


def show_img(img, title="image display", cvt2bgr=False):
    '''Catch-all OpenCV-based image printing function

    :param img: input image. numpy array or tensor of any datatype.
    :param title: title of the displaying window
    :param cvt2bgr: flag to convert image to a grayscale
    :return: N/A
    '''

    if torch.is_tensor(img):  # if this is a tensor, transfer it to the cpu and convert it to a np array
        img = img.cpu().numpy()

    if (img.shape.__len__() == 4):  # this is likely an np array that comes from a tensor representing a batch of images.
        # grab and print the first sample
        img = img[0, :]

    if (img.shape[0] <= 3):  # this is likely a CxHxW representation of an image. turn it to OpenCV's HxWxC
        img = img.transpose(1, 2, 0)

    range = img.max() - img.min()
    if range <= 2:  # image is likely a [0,1] float or a pixel-wise prediction
        print("rescalling the image to [0,255]...")
        img = ((img - img.min()) / (img.max() - img.min())) * 255

    if (img.dtype is not 'uint8'):
        # ('int32' or 'float32'): # 32-bit image. turn to 8-bit before displaying
        img = img.astype(np.uint8)

    if (img.shape.__len__() == 2 or min(img.shape) == 1):  # grayscale image
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    if cvt2bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def filter_detectron2_detections(det, threshold, validRangeOD=None, targetClass=(8, 4)):
    '''
    This function is designed specifically for the way detectron2 organizes its object detection output. It filters
    its bounding boxes based on classes, threshold and valid detection range.

    :param det: output from detectron2's object detection framework
    :param threshold: detection threshold
    :param validRangeOD: valid detection range. It determines a valid range of detections by considering the top-left
    pixel coordinates from the object detection outputs.
        e.g., validRangeOD = [200,600] would only consider bounding boxes whose top-left coordinates sit between rows
        200 and 600. All other bounding boxes are filtered out.
    :param: targetClass: classes you want to consider. Please refer to the list of classes from the COCO dataset. Default:
    8, 4 = boats and airplanes.

    :return: filt_bboxes, filt_scores, filt_class: lists representing, respectively, the filtered bounding boxes, their
    detection scores and classes.
    '''

    filt_bboxes = []
    filt_scores = []
    filt_class = []

    classes = det['instances']._fields['pred_classes']
    scores = det['instances']._fields['scores']
    bboxes = det['instances']._fields['pred_boxes']

    for i in range(0, len(classes)):

        # print("Detection {}: class {} score {}".format(i,classes[i],scores[i]))
        if classes[i] in targetClass and scores[i] >= threshold:
            # get the tensor, transfer it to cpu, turn to numpy array and typecast to int
            bbox = bboxes[i].tensor.cpu().numpy().astype(int)
            classCurrent = classes[i].cpu().numpy().astype(int)

            if (validRangeOD is not None) and (bbox[0][1] < validRangeOD[0] or bbox[0][1] > validRangeOD[1]):
                print('Invalid y-coordinates ({}) on OD! Ignore detection.'.format(bbox[0][1]))
                break

            # get the tensor, transfer it to cpu, turn to numpy array and round it to 2 decimal places
            score = scores[i].cpu().numpy().round(2)
            filt_bboxes.append(bbox)
            filt_scores.append(score)
            filt_class.append(classCurrent)
            # print("Valid! {}".format(bbox))

    return filt_bboxes, filt_scores, filt_class


def concatenate_bbs(input, scores, classes=None, threshold=0.05):
    '''
    Function to concatenate bounding boxes that overlap above a given threshold.
    Caution: Classes of highest-scoring predictions prevail in the concatenated bounding boxes (see illustrative
    image for examples).

    :param input: list or numpy array with each element representing a bounding box.
    :param scores: list of detection scores of each bounding box (see note 1 below)
    :param classes: list with class ID for each bounding box
    :param threshold: amount of IoU between two bounding boxes that would lead to a concatenation.
    :return: concatenated, concatenatedScores, concatenatedClasses: concatenated bounding boxes, scores and classes,
    respectively.

    Note 1: when two bounding boxes are concatenated, the score and class of the highest-scoring one are kept. Therefore,
    this function should be mainly used for the concatenation of bbs OF THE SAME CLASS.

    Note 2:
    Algorithm - the first bbox of the list is compared with each of the subsequent bboxes. if there is an overlap
    larger than the threshold, the two bounding boxes (bbs) are combined. Then the two original bboxes are
    excluded form the list, and the new, concatenated one is placed in the first index of the list.
    When a bbox does not overlap with anyone else, it is added to the "concatenated" list (meaning that there
    is no more combining to be done with it) and excluded from the original list. All bboxes will eventually
    get to this condition, moment when the algorithm stops (because the original list is empty).

    '''

    if classes is None:
        classes = [np.array(8)] * len(scores)

    if isinstance(input, list):
        input = np.asarray(input)

    nboxes = input.__len__()
    boxes = []
    for i in range(nboxes):  # grab all bboxes in a list
        if len(input[0].shape) == 1:
            boxes.append(input[i])
        else:
            boxes.append(input[i][0])

    concatenated = []
    concatenatedScores = []
    concatenatedClasses = []

    # concatenate all the entries that overlap in "boxes"
    while boxes:
        concat = 0  # determines if boxes[0] overlaps with any other BB in the list
        if boxes.__len__() > 1:  # if there are at least two non-processed bboxes, proceed:
            for i in range(1, boxes.__len__()):
                boxA = boxes[0]
                boxB = boxes[i]
                iou = iou_bbs(boxA, boxB) # calculate the IoU between two BBs
                if iou > threshold:
                    # removes the two overlapping BBs from "boxes" (list of unprocessed BBs)
                    boxes.pop(i)
                    boxes.pop(0)

                    # update the score and class of position 0 to reflect the concatenated result
                    if scores[i] > scores[0]:
                        scores[0] = scores[i]
                        scores.pop(i)
                        classes[0] = classes[i]
                        classes.pop(i)
                    else:
                        # in this case, "classes[0]" and "score[0]" do not need to change, but we still need to
                        # exclude scores[i] and classes[i] from the list of BBs
                        scores.pop(i)
                        classes.pop(i)

                    # change the bboxes by the new, concatenated one
                    newBBox = np.array([(min(boxA[0], boxB[0])), (min(boxA[1], boxB[1])), (max(boxA[2], boxB[2])),
                                        (max(boxA[3], boxB[3]))])
                    boxes.insert(0, newBBox)

                    concat = 1
                    break

            # if BB in index 0 did not have any overlap with any other BB,
            # take it out of "boxes" and move it to the final, processed list
            if concat == 0:
                concatenated.append(boxes[0])
                concatenatedScores.append(scores[0])
                concatenatedClasses.append(classes[0])
                boxes.pop(0)
                scores.pop(0)
                classes.pop(0)

        # when there is only one bbox left, add it to the final list and take it out of "boxes" (so that the loop is
        # over)
        else:  # the elements must but taken out of "boxes", otherwise the "while" does not end
            concatenated.append(boxes[0])
            concatenatedScores.append(scores[0])
            concatenatedClasses.append(classes[0])
            boxes.pop(0)
            scores.pop(0)
            classes.pop(0)

    return concatenated, concatenatedScores, concatenatedClasses


def plot_all_bb(img, bb, rand_colors=True, line=1, display=False, score=None, classes=None, title="All Bounding Boxes",
                save_output=False, font_size=1, name="generic_image.jpg"):
    '''
    Function to plot a group of bounding boxes, classes and scores upon an image. The bounding boxes/text can have a
    random color per class, or might all be black. The resulting plotted image might be saved or not.

    :param img: a 3-channel ndarray representing an image
    :param bb: a list where each element represents a (4,) ndarray with the bounding box coordinates
    :param rand_colors: flag to determine if detections from each class will be assigned to a random colors
    :param line: determines the thickness of the bounding box lines
    :param display: flag to determine if the plotted image will be displayed.
    It has to be True (together with save_output) for the outputs to be saved
    :param score: list containing the detection score of each detection (bounding box)
    :param classes: list containing the numerical class associated to each detection
    :param title: title of the plot
    :param save_output: flag to determine if the plotted output will be saved as an images
    :param font_size: determines the size of the font displaying the class and score of each detection
    :param name: if display and save_output are True, this parameter determines the name of the output file.
    :return: ndarray representing the plotted image
    '''
    unique_classes = len(set(classes))
    color = [0] * unique_classes

    for i in range(unique_classes):
        if rand_colors:
            # color[i] = np.random.randint(0,256,3) (this unfortunately does not work, see https://github.com/opencv/opencv/issues/14866)
            color[i] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        else:
            color[i] = (0, 0, 0)

    placeHolderImg = img.copy()

    assert len(classes) == len(bb) == len(score), \
        "The number of scores ({}), classes ({}) and bounding boxes ({}) provided has to be the same.".format(
            len(score), len(classes), len(bb))

    print('Plotting the following bounding boxes')
    for i in range(0, len(bb)):

        color_code = list(set(classes)).index(classes[i])  # grab the unique color code of this class

        if bb[0].shape.__len__() == 2:  # lists of [[np.array]]
            current = bb[i][0].astype(int)
        else:
            current = bb[i].astype(int)

        print('-' * 30)
        print('{}:{},{},{},{}'.format(i, current[0], current[1], current[2], current[3]))
        cv2.rectangle(placeHolderImg, (current[0], current[1]), (current[2], current[3]), color[color_code], line)

        if (score is not None) and (classes is not None):
            cv2.putText(placeHolderImg, (str(round(score[i], 2)) + " C " + str(classes[i])),
                        (current[0] - 10, current[1] - 10), 1,
                        font_size, color[color_code], 1)

    if display:
        cv2.namedWindow(title, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(title, 0, 0)

        if save_output:
            cv2.imwrite(name, placeHolderImg)
            print('Output saved as {}'.format(name))

        cv2.imshow(title, placeHolderImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return placeHolderImg


def read_all_images(dir_path, format='*.jpg', mask_flag=False, return_address=False):
    '''
    This function returns a list where each index represents an image

    :param dir_path: path to the directory to be explored
    :param format: file format to be considered
    :param mask_flag: Flag that determines if an image is a mask (e.g., for semantic and instance segmentation), case in
    which it is read as a 1-channel image.
    :param return_address: Flag to determine if the output will include the address of each image (useful in specific
    applications).
    :return: a list where each element is an ndarray represeting an image. If "return_address" is True, it returns an
    additional list where each element is a string with the address of the equivalent image.
    '''

    images = []
    list = glob.glob(dir_path + format)

    if len(list) is 0:
        raise ValueError("Could not read any file in the directory/format specified.")

    list.sort()  # sort the list

    addresses = []

    for img in list:
        if mask_flag:
            images.append(cv2.imread(img, 0))  # read mask as 1-channel img (for instance/semantic segmentation tasks)
        else:
            addresses.append(img)
            images.append(cv2.imread(img))

    if return_address:
        return [images, addresses]
    else:
        return images


def check_create_dir(path):
    '''
    Function to check if a directory exists. In case it does not, it is created.

    :param path: path of the directory to be checked/created
    :return:
    '''

    if not os.path.isdir(path):
        print('Creating a new folder at {}'.format(path))
        os.makedirs(path)
    else:
        print('Directory {} already exists.'.format(path))


def create_metadata(
        output_add: str = None,
        modelname: str = "unspecified",
        backbone: str = "unspecified",
        model_add: str = "unspecified",
        model_id: str = "ID",
        epochs: int = 0,
        b_size: int = 0,
        lr: float = 0,
        input_size: str = "unspecified",
        optm: str = "unspecified",
        step_size: str = "unspecified",
        gamma: str = "unspecified",
        thresh: float = None,
        AP: float = 0,
        dice: float = 0,
        precision: float = 0,
        recall: float = 0,
        AP_05: float = 0,
        AP_05_095: float = 0,
        notes='unspecified'
):
    '''
    This function creates a simple .txt file to concentrate the metadata of a generic model. Please refer to the names
    of the parameters for a description of their meaning.
    '''

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_filename = now.strftime("%d_%m_%Y %H_%M_%S")

    output_prefix = './'

    if output_add:
        check_create_dir(output_add)
        output_prefix = output_add

    with open(output_prefix + str(model_id) + " [" + dt_filename + ']' + ' metadata.txt', 'w') as file:

        file.write('*' * 20 + ' General info ' + '*' * 20 + '\n')
        file.write('Timestamp: {}\n'.format(dt_string))
        file.write('Model ID: {}\n'.format(model_id))
        file.write('Model name: {}\n'.format(modelname))
        file.write('Model backbone: {}\n'.format(backbone))
        file.write('Saved model address: {}\n'.format(model_add))
        file.write('\n')

        file.write('*' * 20 + ' Hyperparameters ' + '*' * 20 + '\n')
        file.write('Epochs: {}\n'.format(epochs))
        file.write('Batch size: {}\n'.format(b_size))
        file.write('Input sample size: {}\n'.format(input_size))
        file.write('Optimizer: {}\n'.format(optm))
        file.write('Learning rate: {}\n'.format(lr))
        file.write('Step size (in epochs; for the learning scheduler): {}\n'.format(step_size))
        file.write('Gamma (multiplies by the LR every step size): {}\n'.format(gamma))
        file.write('\n')

        file.write('*' * 20 + ' Performance metrics ' + '*' * 20 + '\n')

        if thresh:
            file.write('Detection threshold: {}\n'.format(thresh))
            file.write('Average Precision (AP): {}\n'.format(AP))
            file.write('Precision: {}\n'.format(precision))
            file.write('Recall: {}\n'.format(recall))
            file.write('Dice score: {}\n'.format(dice))

        if AP_05 != "N/A" or AP_05_095 != "N/A":
            file.write('AP @ 0.5: {}\n'.format(AP_05))
            file.write('AP @ [0.5:0.05:0.95]: {}\n'.format(AP_05_095))

        file.write('\n')

        file.write('*' * 20 + ' Additional info ' + '*' * 20 + '\n')
        file.write('{}\n'.format(notes))




