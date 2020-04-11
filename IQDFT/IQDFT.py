#! /usr/bin/env python
import os
import cv2
import dlib
import numpy as np
import scipy.spatial as spatial
import logging

path = os.path.dirname(__file__)
PREDICTOR_PATH = path+r'\models\shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)

FPS = 30 # for videos fps
cropped_id = 0


def extract_faces(file, output_folder=""):
    print("Started to extract faces")
    frame_saved = 0
    video = cv2.VideoCapture(file)
    frames_with_object = []
    while cv2.waitKey(1) < 0:
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break
        frame = frame.copy()

        faces_found = face_detection.face_detection(frame)
        if len(faces_found) == 0:
            continue
        frames_with_object.append(frame)

        frame_saved = frame_saved + 1
        print("Faces found " + str(frame_saved))

        if output_folder != "":
            for face_found in faces_found:
                x1 = face_found.left()
                y1 = face_found.top()
                x2 = face_found.right()
                y2 = face_found.bottom()
                cropped_object = frame[y1:y2, x1:x2]
                cv2.imwrite(output_folder+"/face-" + str(frame_saved) + ".png", cropped_object)
    return frames_with_object


def hide_faces(file, file3="", image_to_add=""):
    print("Started to hide faces")
    frame_saved = 0
    video = cv2.VideoCapture(file)

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    video_writer = cv2.VideoWriter(file3, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS,
                                   (frame_width, frame_height))
    while cv2.waitKey(1) < 0:
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break
        frame = frame.copy()

        faces_found = face_detection.face_detection(frame)
        if len(faces_found) == 0:
            video_writer.write(frame)
            continue

        frame_saved = frame_saved + 1

        face_found = faces_found[0]
        x1 = face_found.left()
        y1 = face_found.top()
        x2 = face_found.right()
        y2 = face_found.bottom()
        if image_to_add != "":
            face_img = frame[y1:y2, x1:x2]
            front_img = cv2.imread(image_to_add)
            front_img = cv2.resize(front_img, (face_img.shape[1], face_img.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            front_img = frame[y1:y2, x1:x2]
            try:
                front_img = cv2.blur(front_img, (50, 50))
            except:
                front_img = cv2.blur(frame, (50, 50))

        frame = add_two_images(frame, front_img, y1, x1, x2, y2)

        print("Fixed frame " + str(frame_saved))
        video_writer.write(frame)
    return "Done"


def apply_faces(file, faces, file3):
    print("Started to apply faces")
    frame_saved = 0
    face_added = 0
    video = cv2.VideoCapture(file)

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    video_writer = cv2.VideoWriter(file3, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS,
                                   (frame_width, frame_height))
    while cv2.waitKey(1) < 0:
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break
        frame = frame.copy()

        faces_found = face_detection.face_detection(frame)
        if len(faces_found) == 0:
            continue

        frame_saved = frame_saved + 1
        face_to_use = faces[face_added]
        face_added = face_added + 1
        if face_added == len(faces):
            face_added = 0

        print("Face swapped " + str(frame_saved))
        new_frame = swap_face(face_to_use, frame)
        try:
            video_writer.write(new_frame)
        except:
            continue
    return "Done"


def add_two_images(back_img, front_img, y1, x1, x2, y2):
    try:
        # I want to put front_img on back_img, So I created a ROI
        rows, cols, channels = front_img.shape
        roi = back_img[0:rows, 0:cols]
        # Now we create a mask of front_img and create its inverse mask also
        img2gray = cv2.cvtColor(front_img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 240, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        # Now black-out the area of front_img in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        # Take only region of front_img from front_img image.
        img2_fg = cv2.bitwise_and(front_img, front_img, mask=mask)
        # Put front_img in ROI and modify the main image
        dst = cv2.add(img1_bg, img2_fg)
        back_img[y1:y2, x1:x2] = dst
    except:
        return back_img
    return back_img


def swap_face(src_img, dst_img):
    correct_color = True
    warp_2d = True

    try:
        # Select src face
        src_points, src_shape, src_face = face_detection.select_face(src_img, predictor)
        # Select dst face
        dst_points, dst_shape, dst_face = face_detection.select_face(dst_img, predictor)

        if src_points is None or dst_points is None:
            print('Detect 0 Face !!!')
            return dst_img
        output = fs.face_swap(src_face, dst_face, src_points, dst_points, dst_shape, dst_img, correct_color, warp_2d)
        return output
    except:
        return dst_img


def swap_faces(file1, file2, file3):
    global cropped_id
    file1 = str.lower(file1)
    file2 = str.lower(file2)
    file3 = str.lower(file3)
    faces = extract_faces(file1)
    apply_faces(file2, faces, file3)
    return "done"


# face swapping class
class fs:

    # 3D Transform
    @staticmethod
    def bilinear_interpolate(img, coords):
        """ Interpolates over every image channel
        http://en.wikipedia.org/wiki/Bilinear_interpolation
        :param img: max 3 channel image
        :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
        :returns: array of interpolated pixels with same shape as coords
        """
        int_coords = np.int32(coords)
        x0, y0 = int_coords
        dx, dy = coords - int_coords

        # 4 Neighour pixels
        q11 = img[y0, x0]
        q21 = img[y0, x0 + 1]
        q12 = img[y0 + 1, x0]
        q22 = img[y0 + 1, x0 + 1]

        btm = q21.T * dx + q11.T * (1 - dx)
        top = q22.T * dx + q12.T * (1 - dx)
        inter_pixel = top * dy + btm * (1 - dy)

        return inter_pixel.T

    @staticmethod
    def grid_coordinates(points):
        """ x,y grid coordinates within the ROI of supplied points
        :param points: points to generate grid coordinates
        :returns: array of (x, y) coordinates
        """
        xmin = np.min(points[:, 0])
        xmax = np.max(points[:, 0]) + 1
        ymin = np.min(points[:, 1])
        ymax = np.max(points[:, 1]) + 1

        return np.asarray([(x, y) for y in range(ymin, ymax)
                           for x in range(xmin, xmax)], np.uint32)

    @staticmethod
    def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
        """
        Warp each triangle from the src_image only within the
        ROI of the destination image (points in dst_points).
        """
        roi_coords = fs.grid_coordinates(dst_points)
        # indices to vertices. -1 if pixel is not in any triangle
        roi_tri_indices = delaunay.find_simplex(roi_coords)

        for simplex_index in range(len(delaunay.simplices)):
            coords = roi_coords[roi_tri_indices == simplex_index]
            num_coords = len(coords)
            out_coords = np.dot(tri_affines[simplex_index],
                                np.vstack((coords.T, np.ones(num_coords))))
            x, y = coords.T
            result_img[y, x] = fs.bilinear_interpolate(src_img, out_coords)

        return None

    @staticmethod
    def triangular_affine_matrices(vertices, src_points, dst_points):
        """
        Calculate the affine transformation matrix for each
        triangle (x,y) vertex from dst_points to src_points
        :param vertices: array of triplet indices to corners of triangle
        :param src_points: array of [x, y] points to landmarks for source image
        :param dst_points: array of [x, y] points to landmarks for destination image
        :returns: 2 x 3 affine matrix transformation for a triangle
        """
        ones = [1, 1, 1]
        for tri_indices in vertices:
            src_tri = np.vstack((src_points[tri_indices, :].T, ones))
            dst_tri = np.vstack((dst_points[tri_indices, :].T, ones))
            mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
            yield mat

    @staticmethod
    def warp_image_3d(src_img, src_points, dst_points, dst_shape, dtype=np.uint8):
        rows, cols = dst_shape[:2]
        result_img = np.zeros((rows, cols, 3), dtype=dtype)

        delaunay = spatial.Delaunay(dst_points)
        tri_affines = np.asarray(list(fs.triangular_affine_matrices(
            delaunay.simplices, src_points, dst_points)))

        fs.process_warp(src_img, result_img, tri_affines, dst_points, delaunay)

        return result_img

    # 2D Transform
    @staticmethod
    def transformation_from_points(points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(np.dot(points1.T, points2))
        R = (np.dot(U, Vt)).T

        return np.vstack([np.hstack([s2 / s1 * R,
                                     (c2.T - np.dot(s2 / s1 * R, c1.T))[:, np.newaxis]]),
                          np.array([[0., 0., 1.]])])

    @staticmethod
    def warp_image_2d(im, M, dshape):
        output_im = np.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(im,
                       M[:2],
                       (dshape[1], dshape[0]),
                       dst=output_im,
                       borderMode=cv2.BORDER_TRANSPARENT,
                       flags=cv2.WARP_INVERSE_MAP)

        return output_im

    # Generate Mask
    @staticmethod
    def mask_from_points(size, points, erode_flag=1):
        radius = 10  # kernel size
        kernel = np.ones((radius, radius), np.uint8)

        mask = np.zeros(size, np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
        if erode_flag:
            mask = cv2.erode(mask, kernel, iterations=1)

        return mask

    # Color Correction
    @staticmethod
    def correct_colours(im1, im2, landmarks1):
        COLOUR_CORRECT_BLUR_FRAC = 0.75
        LEFT_EYE_POINTS = list(range(42, 48))
        RIGHT_EYE_POINTS = list(range(36, 42))

        blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
            np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
            np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

        # Avoid divide-by-zero errors.
        im2_blur = im2_blur.astype(int)
        im2_blur += 128 * (im2_blur <= 1)

        result = im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    # Copy-and-paste
    @staticmethod
    def apply_mask(img, mask):
        """ Apply mask to supplied image
        :param img: max 3 channel image
        :param mask: [0-255] values in mask
        :returns: new image with mask applied
        """
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        return masked_img

    # Alpha blending
    @staticmethod
    def alpha_feathering(src_img, dest_img, img_mask, blur_radius=15):
        mask = cv2.blur(img_mask, (blur_radius, blur_radius))
        mask = mask / 255.0

        result_img = np.empty(src_img.shape, np.uint8)
        for i in range(3):
            result_img[..., i] = src_img[..., i] * mask + dest_img[..., i] * (1 - mask)

        return result_img

    @staticmethod
    def check_points(img, points):
        # Todo: I just consider one situation.
        if points[8, 1] > img.shape[0]:
            logging.error("Jaw part out of image")
        else:
            return True
        return False

    @staticmethod
    def face_swap(src_face, dst_face, src_points, dst_points, dst_shape, dst_img, correct_color, warp_2d, end=48):
        h, w = dst_face.shape[:2]

        # 3d warp
        warped_src_face = fs.warp_image_3d(src_face, src_points[:end], dst_points[:end], (h, w))
        # Mask for blending
        mask = fs.mask_from_points((h, w), dst_points)
        mask_src = np.mean(warped_src_face, axis=2) > 0
        mask = np.asarray(mask * mask_src, dtype=np.uint8)
        # Correct color
        if correct_color:
            warped_src_face = fs.apply_mask(warped_src_face, mask)
            dst_face_masked = fs.apply_mask(dst_face, mask)
            warped_src_face = fs.correct_colours(dst_face_masked, warped_src_face, dst_points)
        # 2d warp
        if warp_2d:
            unwarped_src_face = fs.warp_image_3d(warped_src_face, dst_points[:end], src_points[:end], src_face.shape[:2])
            warped_src_face = fs.warp_image_2d(unwarped_src_face, fs.transformation_from_points(dst_points, src_points),
                                            (h, w, 3))

            mask = fs.mask_from_points((h, w), dst_points)
            mask_src = np.mean(warped_src_face, axis=2) > 0
            mask = np.asarray(mask * mask_src, dtype=np.uint8)

        # Shrink the mask
        kernel = np.ones((10, 10), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        # Poisson Blending
        r = cv2.boundingRect(mask)
        center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
        output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)

        x, y, w, h = dst_shape
        dst_img_cp = dst_img.copy()
        dst_img_cp[y:y + h, x:x + w] = output

        return dst_img_cp


# face detection class
class face_detection:

    # Face detection

    @staticmethod
    def face_detection(img, upsample_times=1):
        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        detector = dlib.get_frontal_face_detector()
        faces = detector(img, upsample_times)

        return faces

    # PREDICTOR_PATH = 'models/shape_predictor_68_face_landmarks.dat'
    # predictor = dlib.shape_predictor(PREDICTOR_PATH)

    # Face and points detection
    @staticmethod
    def face_points_detection(img, bbox: dlib.rectangle, predictor):
        try:
            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, bbox)

            # loop over the 68 facial landmarks and convert them
            # to a 2-tuple of (x, y)-coordinates
            coords = np.asarray(list([p.x, p.y] for p in shape.parts()), dtype=np.int)

            # return the array of (x, y)-coordinates
            return coords
        except:
            return None

    @staticmethod
    def select_face(im, predictor, r=10, choose=True):
        faces = face_detection.face_detection(im)

        if len(faces) == 0:
            return None, None, None

        bbox = []
        if len(faces) == 1 or not choose:
            idx = np.argmax([(face.right() - face.left()) * (face.bottom() - face.top()) for face in faces])
            try:
                bbox = faces[idx]
            except:
                pass

            # we use the first face we detect
            bbox.append(faces[0])
            bbox = bbox[0]

        points = np.asarray(face_detection.face_points_detection(im, bbox, predictor))
        if points is None:
            return None, None, None

        try:
            im_w, im_h = im.shape[:2]
            left, top = np.min(points, 0)
            right, bottom = np.max(points, 0)

            x, y = max(0, left - r), max(0, top - r)
            w, h = min(right + r, im_h) - x, min(bottom + r, im_w) - y

            return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y + h, x:x + w]
        except:
            return None, None, None







