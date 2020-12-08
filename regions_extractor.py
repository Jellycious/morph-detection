#!/home/jelle/.virtualenvs/face-morphing/bin/python
# This makes sure that it runs the virtual env

# third party libraries
import dlib
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# standard libraries
import os
import sys
import pickle

# our libraries
import utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./predictors/shape_predictor_68_face_landmarks.dat')

win = dlib.image_window()


def main():
    for (trusted, questioned) in utils.pairs_iter():
        show_extracted_regions(trusted)

def save_fft_example_image():
    for (trusted, questioned) in utils.pairs_iter():
        dirn = os.path.dirname(trusted)
        for (prefix, trusted, questioned) in utils.get_regions(dirn):
            region_img = cv2.imread(trusted, 0)
            region_fft = np.fft.fftshift(np.fft.fft2(region_img))
            region_fft_img = 20 * np.log(np.abs(region_fft))
            print(trusted)

            cv2.imwrite('fft_region_example.png', region_fft_img)
            return


def check_bona_fide_of_directory(dirn):
    if os.path.split(dirn)[1][0] == 'M':
        bona_fide = False
        return bona_fide
    elif os.path.split(dirn)[1][0] == 'B':
        bona_fide = True
        return bona_fide
    else:
        raise Exception("Could not determine class of: " + dirn)



def create_frequency_training_data():

    frequency_data_list = list()

    for (trusted, questioned) in utils.pairs_iter():

        # dirn = os.path.join(os.path.dirname(trusted), '33x33')
        dirn = os.path.dirname(trusted)

        region_dict = dict()
        for (prefix, trusted, questioned) in utils.get_regions(dirn):

            try:
                trusted_img = cv2.imread(trusted, 0)
                trusted_fft = np.fft.fft2(trusted_img)
                trusted_fft = np.fft.fftshift(trusted_fft)

                questioned_img = cv2.imread(questioned, 0)
                questioned_fft = np.fft.fft2(questioned_img)
                questioned_fft = np.fft.fftshift(questioned_fft)

                region_dict[prefix] = (trusted_fft, questioned_fft)

            except Exception as e:
                print(e)
                print(trusted)

        bona_fide = True
        if os.path.split(dirn)[1][0] == 'M':
            bona_fide = False
        elif os.path.split(dirn)[1][0] == 'B':
            bona_fide = True
        else:
            raise Exception("Could not determine class of: " + dirn)

        frequency_data_list.append((region_dict, bona_fide, os.path.split(dirn)[1]))

    spliced = train_test_split(frequency_data_list, train_size=0.8)
    train_test_frequency_data = {'training': spliced[0], 'test': spliced[1]}

    with open("train_test_frequency_data_RAW.pk", "wb") as output_file:
        pickle.dump(train_test_frequency_data, output_file)


def create_spectral_training_data():

    spectral_data_list = list()

    for (trusted, questioned) in utils.pairs_iter():

        # dirn = os.path.join(os.path.dirname(trusted), '33x33')
        dirn = os.path.dirname(trusted)

        region_dict = dict()
        for (prefix, trusted, questioned) in utils.get_regions(dirn):

            try:
                trusted_img = cv2.imread(trusted, 0)
                trusted_spec = get_spectrum_img_32(trusted_img, replace_negatives=False, log=False)

                questioned_img = cv2.imread(questioned, 0)
                questioned_spec = get_spectrum_img_32(questioned_img, replace_negatives=False, log=False)

                region_dict[prefix] = (trusted_spec, questioned_spec)

            except Exception as e:
                print(e)
                print(trusted)

        bona_fide = True
        if os.path.split(dirn)[1][0] == 'M':
            bona_fide = False
        elif os.path.split(dirn)[1][0] == 'B':
            bona_fide = True
        else:
            raise Exception("Could not determine class of: " + dirn)

        spectral_data_list.append((region_dict, bona_fide, os.path.split(dirn)[1]))

    spliced = train_test_split(spectral_data_list, train_size=0.8)
    train_test_spectral_data = {'training': spliced[0], 'test': spliced[1]}

    with open("train_test_spectral_data_RAW_nolog.pk", "wb") as output_file:
        pickle.dump(train_test_spectral_data, output_file)


def get_spectrum_img_32(img, replace_negatives=True, log=True):
    '''
    EXPECTS AN IMAGE TO BE 32x32 in size
    '''

    fft = np.fft.fft2(img)
    fft = np.fft.fftshift(fft)

    magnitude = np.abs(fft)

    if log:
        magnitude = 20 * np.log(magnitude)

    if replace_negatives:
        magnitude[magnitude < 0] = 0

    CENTER = (16, 16)  # since image is 32x32
    MAX_RADIUS = 16

    warped = cv2.warpPolar(magnitude, (0, 0), CENTER, MAX_RADIUS, cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS)
    warped_transposed = np.transpose(warped)

    spectrum = list()
    for n in warped_transposed:
        avg = np.average(n)
        spectrum.append(avg)

    return spectrum


def increase_image_size(img, scaling):
    height = len(img)
    width = len(img[0])
    return cv2.resize(img, (int(width * scaling), int(height * scaling)), interpolation=cv2.INTER_NEAREST)


def complex_to_polar2(carr):
    '''
    Converts complex numbers into polar coordinates.
    Since complex numbers are a single datatype this will increase the dimension of the matrix by 1.
    The output of a complex number c will be the tuple (distance, angle)
    '''
    # probably this function is useless
    distance = np.abs(carr)
    angle = np.angle(carr)
    return np.dstack((distance, angle))


def frequency_domain_to_polar_frequency_domain(parr):
    '''
    converts frequency domain represented by (hF, vF) to (D, angle)
    :param parr: polar coordinate matrix.
    '''
    return


def extract_regions(imgpath):
    '''
    Extracts the different regions and stores them in the directory.
    Trusted: whether the given image is a trusted image
    '''
    # create file path
    basename = os.path.basename(imgpath)
    dirn = os.path.dirname(imgpath)

    # Check whether image is trusted or questioned
    if basename.startswith('Q'):
        imgcat = 'Q'
    elif basename.startswith('T'):
        imgcat = 'T'
    else:
        raise Exception("Could not determine img state")

    # load image in BGR format
    img = cv2.imread(imgpath)

    first_face_bounding_box = detector(img, 0)[0]
    shape = predictor(img, first_face_bounding_box)

    # get all regions
    nose_region = create_rectangle(shape.parts()[29], 32)
    nose_region_img = extract_rectangle_from_img(img, nose_region)
    cv2.imwrite(os.path.join(dirn, 'NO_' + imgcat + '.png'), nose_region_img)

    right_cheekbone_region = create_rectangle(right_cheekbone_point(shape), 32)
    right_cheekbone_region_img = extract_rectangle_from_img(img, right_cheekbone_region)
    cv2.imwrite(os.path.join(dirn, 'RB_' + imgcat + '.png'), right_cheekbone_region_img)

    left_cheekbone_region = create_rectangle(left_cheekbone_point(shape), 32)
    left_cheekbone_region_img = extract_rectangle_from_img(img, left_cheekbone_region)
    cv2.imwrite(os.path.join(dirn, 'LB_' + imgcat + '.png'), left_cheekbone_region_img)

    right_cheek_region = create_rectangle(right_cheek_point(shape), 32)
    right_cheek_region_img = extract_rectangle_from_img(img, right_cheek_region)
    cv2.imwrite(os.path.join(dirn, 'RC_' + imgcat + '.png'), right_cheek_region_img)

    left_cheek_region = create_rectangle(left_cheek_point(shape), 32)
    left_cheek_region_img = extract_rectangle_from_img(img, left_cheek_region)
    cv2.imwrite(os.path.join(dirn, 'LC_' + imgcat + '.png'), left_cheek_region_img)

    forehead_region = create_rectangle(forehead_point(shape), 32)
    forehead_region_img = extract_rectangle_from_img(img, forehead_region)
    cv2.imwrite(os.path.join(dirn, 'FH_' + imgcat + '.png'), forehead_region_img)

    chin_region = create_rectangle(chin_point(shape), 32)
    chin_region_img = extract_rectangle_from_img(img, chin_region)
    cv2.imwrite(os.path.join(dirn, 'CH_' + imgcat + '.png'), chin_region_img)

    return


def extract_regions(imgpath, size):
    '''
    Extracts regions with dimensions: size x size.
    The results are stored in a directory named "'size'x'size'"
    '''

    # create file path
    basename = os.path.basename(imgpath)
    dirn = os.path.dirname(imgpath)

    # Check whether image is trusted or questioned
    if basename.startswith('Q'):
        imgcat = 'Q'
    elif basename.startswith('T'):
        imgcat = 'T'
    else:
        raise Exception("Could not determine img state")

    # create directory
    new_dir = os.path.join(dirn, '{}x{}'.format(size, size))

    try:
        os.mkdir(new_dir)
    except FileExistsError:
        pass
    except Exception as e:
        print(e)

    # load image in BGR format
    img = cv2.imread(imgpath)

    first_face_bounding_box = detector(img, 0)[0]
    shape = predictor(img, first_face_bounding_box)

    # get all regions
    nose_region = create_rectangle(shape.parts()[29], size)
    nose_region_img = extract_rectangle_from_img(img, nose_region)
    cv2.imwrite(os.path.join(new_dir, 'NO_' + imgcat + '.png'), nose_region_img)

    right_cheekbone_region = create_rectangle(right_cheekbone_point(shape), size)
    right_cheekbone_region_img = extract_rectangle_from_img(img, right_cheekbone_region)
    cv2.imwrite(os.path.join(new_dir, 'RB_' + imgcat + '.png'), right_cheekbone_region_img)

    left_cheekbone_region = create_rectangle(left_cheekbone_point(shape), size)
    left_cheekbone_region_img = extract_rectangle_from_img(img, left_cheekbone_region)
    cv2.imwrite(os.path.join(new_dir, 'LB_' + imgcat + '.png'), left_cheekbone_region_img)

    right_cheek_region = create_rectangle(right_cheek_point(shape), size)
    right_cheek_region_img = extract_rectangle_from_img(img, right_cheek_region)
    cv2.imwrite(os.path.join(new_dir, 'RC_' + imgcat + '.png'), right_cheek_region_img)

    left_cheek_region = create_rectangle(left_cheek_point(shape), size)
    left_cheek_region_img = extract_rectangle_from_img(img, left_cheek_region)
    cv2.imwrite(os.path.join(new_dir, 'LC_' + imgcat + '.png'), left_cheek_region_img)

    forehead_region = create_rectangle(forehead_point(shape), size)
    forehead_region_img = extract_rectangle_from_img(img, forehead_region)
    cv2.imwrite(os.path.join(new_dir, 'FH_' + imgcat + '.png'), forehead_region_img)

    chin_region = create_rectangle(chin_point(shape), size)
    chin_region_img = extract_rectangle_from_img(img, chin_region)
    cv2.imwrite(os.path.join(new_dir, 'CH_' + imgcat + '.png'), chin_region_img)

    return


def show_extracted_regions(imgpath):

    win = dlib.image_window()

    # load grayscale img
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    win.clear_overlay()
    win.set_image(img)

    # detect face
    first_face_bounding_box = detector(img, 0)[0]
    shape = predictor(img, first_face_bounding_box)

    # nose region
    #win.add_overlay(create_rectangle(shape.parts()[29], 32), dlib.rgb_pixel(255, 0, 0))

    #right_cheekbone_center = right_cheekbone_point(shape)
    #win.add_overlay(create_rectangle(right_cheekbone_center, 32), dlib.rgb_pixel(0, 255, 0))

    #left_cheekbone_center = left_cheekbone_point(shape)
    #win.add_overlay(create_rectangle(left_cheekbone_center, 32), dlib.rgb_pixel(0, 0, 255))

    #right_cheek_center = right_cheek_point(shape)
    #win.add_overlay(create_rectangle(right_cheek_center, 32), dlib.rgb_pixel(255, 255, 0))

    #left_cheek_center = left_cheek_point(shape)
    #win.add_overlay(create_rectangle(left_cheek_center, 32), dlib.rgb_pixel(255, 0, 255))

    #forehead = forehead_point(shape)
    #win.add_overlay(create_rectangle(forehead, 32), dlib.rgb_pixel(0, 255, 255))

    #chin = chin_point(shape)
    #win.add_overlay(create_rectangle(chin, 32), dlib.rgb_pixel(255, 255, 255))
    draw_shape_points(shape, win)

    #wait_for_next_input(win)

    #extr_img = extract_rectangle_from_img(img, create_rectangle(shape.parts()[29], 32))

    #win.clear_overlay()
    #win.set_image(extr_img)

    wait_for_next_input(win)

    win.clear_overlay()


def extract_rectangle_from_img(img, rectangle):
    '''
    Returns new img with content of the rectangle on the image.
    '''
    tl = rectangle.tl_corner()
    tr = rectangle.tr_corner()
    bl = rectangle.bl_corner()

    return img[tl.y:bl.y, tl.x:tr.x]


def right_cheekbone_point(shape):
    '''
    Gets the right cheekbone point.
    Shape needs to be shape from dlib 68 landmark predictor.
    '''
    points = shape.parts()
    # cheekbone points
    middle_cheekbone_point = points[29]
    right_cheekbone_upper = points[15]
    right_cheekbone_lower = points[14]

    x = int(0.33 * middle_cheekbone_point.x + 0.33 * right_cheekbone_upper.x + 0.33 * right_cheekbone_lower.x)
    y = int(0.33 * middle_cheekbone_point.y + 0.33 * right_cheekbone_upper.y + 0.33 * right_cheekbone_lower.y)
    right_cheekbone_center = dlib.point(x, y)
    return right_cheekbone_center


def left_cheekbone_point(shape):
    '''
    Gets the left cheekbone point.
    Shape needs to be shape from dlib 68 landmark predictor.
    '''
    # cheekbone points
    points = shape.parts()
    middle_cheekbone_point = points[29]
    left_cheekbone_upper = points[1]
    left_cheekbone_lower = points[2]

    x = int(0.33 * middle_cheekbone_point.x + 0.33 * left_cheekbone_upper.x + 0.33 * left_cheekbone_lower.x)
    y = int(0.33 * middle_cheekbone_point.y + 0.33 * left_cheekbone_upper.y + 0.33 * left_cheekbone_lower.y)
    left_cheekbone_center = dlib.point(x, y)
    return left_cheekbone_center


def left_cheek_point(shape):
    '''
    Left point of cheek. Close to the corner of the mouth.
    Shape needs to be from dlib 68 landmark predictor.
    '''

    points = shape.parts()
    left_mouth_point = points[48]
    left_cheek_upper = points[3]
    left_cheek_lower = points[4]

    x = int(0.5 * left_mouth_point.x + 0.25 * left_cheek_upper.x + 0.25 * left_cheek_lower.x)
    y = int(0.5 * left_mouth_point.y + 0.25 * left_cheek_upper.y + 0.25 * left_cheek_lower.y)
    left_cheek_center = dlib.point(x, y)
    return left_cheek_center


def right_cheek_point(shape):
    '''
    Right point of cheek. Close to the corner of the mouth.
    Shape needs to be from dlib 68 landmark predictor.
    '''

    points = shape.parts()
    right_mouth_point = points[54]
    right_cheek_upper = points[13]
    right_cheek_lower = points[12]
    # right cheek region
    x = int(0.5 * right_mouth_point.x + 0.25 * right_cheek_upper.x + 0.25 * right_cheek_lower.x)
    y = int(0.5 * right_mouth_point.y + 0.25 * right_cheek_upper.y + 0.25 * right_cheek_lower.y)
    right_cheek_center = dlib.point(x, y)
    return right_cheek_center


def forehead_point(shape):
    '''
    Forehead point.
    '''

    points = shape.parts()
    left = points[19]
    right = points[24]
    nose_upper = points[27]

    x = int(0.5 * left.x + 0.5 * right.x)
    y = int(0.5 * left.y + 0.5 * right.y)
    y_change = int(0.5 * (nose_upper.y - y))
    forehead_point = dlib.point(x, y - y_change)
    return forehead_point


def draw_shape_points(shape, win):
    for point in shape.parts():
        win.add_overlay_circle(point, 2, dlib.rgb_pixel(0, 255, 0))


def chin_point(shape):
    '''
    Chin point
    '''

    points = shape.parts()
    bottom_mouth = points[57]
    right_chin = points[9]
    left_chin = points[7]

    x = int(0.5 * bottom_mouth.x + 0.25 * right_chin.x + 0.25 * left_chin.x)
    y = int(0.5 * bottom_mouth.y + 0.25 * right_chin.y + 0.25 * left_chin.y)
    chin_point = dlib.point(x, y)
    return chin_point


def create_rectangle(center, size):
    if (size % 2) == 0:
        s = size / 2
        x1 = int(center.x - s)
        x2 = int(center.x + s)
        y1 = int(center.y - s)
        y2 = int(center.y + s)
    else:
        # since we are dealing with a discrete space.
        # the solution is to offset the region to the right and top with 1 pixel.
        # not ideal, but this should not matter, since this is done for all images.
        s = int(size / 2)
        x1 = int(center.x - s) - 1
        x2 = int(center.x + s)
        y1 = int(center.y - s) - 1
        y2 = int(center.y + s)

    return dlib.rectangle(x1, y1, x2, y2)


def wait_for_next_input(win):
    '''
    Wait untill the user presses a key.
    If the key is 'x' exit.
    '''
    keypress = win.get_next_keypress()

    if (keypress) == 'x':
        sys.exit(0)


if __name__ == '__main__':
    main()
