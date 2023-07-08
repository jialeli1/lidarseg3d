from copy import deepcopy

# import mmcv
import numpy as np
# from mmcv.utils import deprecated_api_warning, is_tuple_of
# from numpy import random
import cv2

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def image_input_transform(input_image, mean, std):
    """
    input_image: [h,w,3]
    mean: [1,1,3]
    std: [1,1,3]
    """
    image = input_image.astype(np.float32)
    image = image / 255.0 

    image -= mean
    image /= std
    return image



def jpeg_compression(image, quality_noise=[30, 70], probability=0.5):
    """
    image: [h, w, 3], bgr mode in cv2
    """
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability]
    )

    if enable:
        if not isinstance(quality_noise, list):
            quality_noise = [-quality_noise, quality_noise]

        quality = int(np.random.uniform(quality_noise[0], quality_noise[1]))

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        # encode
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        # decode
        image = cv2.imdecode(encimg, 1)

    return image



def image_and_points_cp_and_label_random_horizon_flip(image, points_cp, image_label=None, probability=0.5):
    """
    image: [h, w, 3]
    points_cp: [:, ]
    """
    height, width, channel = image.shape

    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability]
    )
    if enable:
        image = np.flip(image, axis=1)
        points_cp = (width - 1) - points_cp
        
        if image_label is not None:
            image_label = np.flip(image_label, axis=1)

    return image, points_cp, image_label



def image_and_points_cp_and_label_resize(image, points_cp, image_label=None, resized_shape=None):
    """
    NOTE: Handles only a single local camera

    image: (H, W, 3)
    points_cp: (N, 3), 3: [cam_id, idx_of_width, idx_of_height]
    resized_shape: (W, H) in cv2 mode
    """
    ori_shape = image.shape # (1280, 1920, 3) or (886, 1920, 3)
    resized_image = cv2.resize(image, resized_shape) 
    res_shape = resized_image.shape # like (640, 960, 3)

    width_ratio = float(res_shape[1]) / float(ori_shape[1]) 
    height_ratio = float(res_shape[0]) / float(ori_shape[0]) 

    points_cp[:, 1] = points_cp[:, 1] * width_ratio
    points_cp[:, 2] = points_cp[:, 2] * height_ratio

    if image_label is not None:
        image_label = cv2.resize(image_label, resized_shape, interpolation=cv2.INTER_NEAREST)
    
    return resized_image, points_cp, image_label


class RandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=0.75, kept_min_ratio=0.75, ignore_index=0, unvalid_cam_id=0, try_times=10, **kwargs):
        # NOTE: crop_size should be (h, w)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.try_times = try_times
        self.cat_max_ratio = cat_max_ratio 
        self.kept_min_ratio = kept_min_ratio 
        self.ignore_index = ignore_index
        self.unvalid_cam_id = unvalid_cam_id

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def crop_points_cp(self, points_cp, crop_bbox):
        """
        Crop from ``points_cp``
        points_cp: (N, 3), 3: [cam_id, idx_of_width, idx_of_height], NOTE!
        """
        # the new coordinate origin should be (crop_y1, crop_x1)
        # NOTE: be carefully
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        points_cp[:, 1] = points_cp[:, 1] - crop_x1
        points_cp[:, 2] = points_cp[:, 2] - crop_y1

        crop_height = self.crop_size[0]
        crop_width = self.crop_size[1]

        # width coord check
        out_of_range_flag_wid = (points_cp[:, 1] < 0) | (points_cp[:, 1] >= crop_width)
        # height coord check
        out_of_range_flag_hei = (points_cp[:, 2] < 0) | (points_cp[:, 2] >= crop_height)
        out_of_range_flag = out_of_range_flag_wid | out_of_range_flag_hei
        

        # set cam_id as 0 for points out of range
        points_cp[out_of_range_flag, 0] = self.unvalid_cam_id
        

        return points_cp


    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.


        Args:
            results (dict): Result dict from loading pipeline.
            image: (Hi, Wi, 3)
            points_cp: (N, 3), [cam_id, idx_of_width, idx_of_height], cam_id: 0-num_cams, 0 is unvalid
            image_label: (Hi, Wi)

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.

        """
        img = results["image"]
        points_cp = results.get("points_cp", None)
        img_label = results.get("image_label", None)
        img_has_annotations = img_label is not None

        crop_valid = False
        crop_bbox = self.get_crop_bbox(img)
        if (self.cat_max_ratio < 1.) and (self.kept_min_ratio > 0.):
            # Repeat 10 times
            for _ in range(self.try_times):
                points_cp_temp = self.crop_points_cp(deepcopy(points_cp), crop_bbox)
                num_points = points_cp_temp.shape[0]
                num_points_inside = (points_cp_temp[:, 0] > 0).sum()
                points_cp_valid = num_points_inside / num_points >= self.kept_min_ratio
                if img_has_annotations:
                    seg_temp = self.crop(deepcopy(img_label), crop_bbox)
                    labels, cnt = np.unique(seg_temp, return_counts=True)
                    cnt = cnt[labels != self.ignore_index]
                    label_valid = len(cnt) > 1 and (np.max(cnt) / np.sum(cnt)) < self.cat_max_ratio
                else: 
                    label_valid = True


                if points_cp_valid & label_valid:
                    crop_valid = True
                    valid_crop_bbox = crop_bbox
                    break
                crop_bbox = self.get_crop_bbox(img)


        results['crop_valid'] = crop_valid
        if crop_valid:
            # crop the image using final crop_bbox
            img = self.crop(img, valid_crop_bbox)
            img_shape = img.shape
            results['image'] = img
            results['cropped_shape'] = img_shape

            # crop the points_cp using final crop_bbox
            points_cp = self.crop_points_cp(points_cp, valid_crop_bbox)
            results['points_cp'] = points_cp

            # crop the image_label using final crop_bbox
            if img_has_annotations:
                img_label = self.crop(img_label, valid_crop_bbox)
                results['image_label'] = img_label
        else:
            img, points_cp, img_label = image_and_points_cp_and_label_resize(
                image=img, 
                points_cp=points_cp, 
                image_label=img_label, 
                resized_shape=(self.crop_size[1], self.crop_size[0]), # cv2 format (W, H)
            )
            img_shape = img.shape

            results['image'] = img
            results['cropped_shape'] = img_shape
            results['points_cp'] = points_cp
            if img_has_annotations:
                results['image_label'] = img_label


        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'



class RandomRescale(object):
    """Resize image & points_cp & image_label in a local camera.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
            Default:None. (W, H) mode
        ratio_range (tuple[float]): (min_ratio, max_ratio).
            Default: None
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Default: True
    """

    def __init__(self,
                 img_scale=None,
                 ratio_range=None,
                 keep_ratio=True,
                 ):
        assert ratio_range is not None

        self.img_scale = img_scale
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio


    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)

        return scale, None



    def _random_scale(self, results):
        if self.img_scale is None:
            h, w = results['image'].shape[:2]
            scale, _ = self.random_sample_ratio((w, h), self.ratio_range)
        else:
            # (hi, wi, 3)
            assert results['image'].shape[0] == self.img_scale[1]
            assert results['image'].shape[1] == self.img_scale[0]

            scale, _ = self.random_sample_ratio(self.img_scale, self.ratio_range)

        results['scale'] = scale


    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.
            image: (Hi, Wi, 3)
            points_cp: (N, 3), [cam_id, idx_of_width, idx_of_height], cam_id: 0-num_cams, 0 is unvalid
            image_label: (Hi, Wi)

        Returns:
            dict: Resized results
        """

        if 'scale' not in results:
            self._random_scale(results)

        random_scale = results["scale"]
        img = results["image"]
        points_cp = results.get("points_cp", None)
        img_label = results.get("image_label", None)
        img_has_annotations = img_label is not None

        img, points_cp, img_label = image_and_points_cp_and_label_resize(
            image=img, 
            points_cp=points_cp, 
            image_label=img_label, 
            resized_shape=random_scale, 
        )
        img_shape = img.shape

        results['image'] = img
        results['rescaled_shape'] = img_shape # (H, W, 3)
        results['points_cp'] = points_cp
        if img_has_annotations:
            results['image_label'] = img_label


        return results



class RandomRotate(object):
    """Rotate the image & points_cp & image_label.
    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """

    def __init__(self,
                 degree,
                 pad_val=0,
                 seg_pad_val=255, 
                 center=None,
                 auto_bound=False,
                 crop_size=None, 
                 cat_max_ratio=0.75,
                 kept_min_ratio=0.60,
                 ignore_index=0, 
                 unvalid_cam_id=0, 
                 try_times=10):
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        self.pal_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound


        # for crop
        self.crop_size = crop_size
        self.try_times = try_times
        self.cat_max_ratio = cat_max_ratio  
        self.kept_min_ratio = kept_min_ratio  
        self.ignore_index = ignore_index
        self.unvalid_cam_id = unvalid_cam_id


    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2


    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img


    def crop_points_cp(self, points_cp, crop_bbox):
        """
        Crop from ``points_cp``
        points_cp: (N, 3), 3: [cam_id, idx_of_width, idx_of_height], NOTE!
        """
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        points_cp[:, 1] = points_cp[:, 1] - crop_x1
        points_cp[:, 2] = points_cp[:, 2] - crop_y1

        crop_height = self.crop_size[0]
        crop_width = self.crop_size[1]

        # width coord check
        out_of_range_flag_wid = (points_cp[:, 1] < 0) | (points_cp[:, 1] >= crop_width)
        # height coord check
        out_of_range_flag_hei = (points_cp[:, 2] < 0) | (points_cp[:, 2] >= crop_height)
        out_of_range_flag = out_of_range_flag_wid | out_of_range_flag_hei
        

        points_cp[out_of_range_flag, 0] = self.unvalid_cam_id
        
        return points_cp


    def rotate_image(self, 
                img,
                angle,
                center=None,
                scale=1.0,
                border_value=0,
                interpolation='bilinear',
                auto_bound=False):
        """Rotate an image, modified from mmcv

        Args:
            img (ndarray): Image to be rotated.
            angle (float): Rotation angle in degrees, positive values mean
                clockwise rotation.
            center (tuple[float], optional): Center point (w, h) of the rotation in
                the source image. If not specified, the center of the image will be
                used.
            scale (float): Isotropic scale factor.
            border_value (int): Border value.
            interpolation (str): Same as :func:`resize`.
            auto_bound (bool): Whether to adjust the image size to cover the whole
                rotated image.

        Returns:
            ndarray: The rotated image.
        """
        if center is not None and auto_bound:
            raise ValueError('`auto_bound` conflicts with `center`')
        h, w = img.shape[:2]
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        assert isinstance(center, tuple)

        matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        if auto_bound:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = h * sin + w * cos
            new_h = h * cos + w * sin
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = int(np.round(new_w))
            h = int(np.round(new_h))
        rotated = cv2.warpAffine(
            img,
            matrix, (w, h),
            flags=cv2_interp_codes[interpolation],
            borderValue=border_value)
        return rotated


    def rotate_points_cp(self, 
                img,
                points_cp,
                angle,
                center=None,
                scale=1.0,
                border_value=0,
                interpolation='bilinear',
                auto_bound=False):
        """Rotate an image, modified from mmcv

        Args:
            img (ndarray): rotated img for points_cp check!
            points_cp (ndarray): (N, 3), cam_id, wid_coord, hei_coord 
            angle (float): Rotation angle in degrees, positive values mean
                clockwise rotation.
            center (tuple[float], optional): Center point (w, h) of the rotation in
                the source image. If not specified, the center of the image will be
                used.
            scale (float): Isotropic scale factor.
            border_value (int): Border value.
            interpolation (str): Same as :func:`resize`.
            auto_bound (bool): Whether to adjust the image size to cover the whole
                rotated image.

        Returns:
            ndarray: The rotated image.
        """
        if center is not None and auto_bound:
            raise ValueError('`auto_bound` conflicts with `center`')
        h, w = img.shape[:2]
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)
        assert isinstance(center, tuple)

        matrix = cv2.getRotationMatrix2D(center, -angle, scale)
        if auto_bound:
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            new_w = h * sin + w * cos
            new_h = h * cos + w * sin
            matrix[0, 2] += (new_w - w) * 0.5
            matrix[1, 2] += (new_h - h) * 0.5
            w = int(np.round(new_w))
            h = int(np.round(new_h))

        points_cam_coord = points_cp[:, 0]
        points_wid_coord = points_cp[:, 1]
        points_hei_coord = points_cp[:, 2]
        ones = np.ones_like(points_wid_coord)

        # NOTE: coord in CV2 mode!
        # [N, ] -> [N, 3]
        point_hom_coords = np.stack([points_wid_coord, points_hei_coord, ones], axis=1)
        # [2, 3] matmul [3, N] -> [2, N] -> [N, 2]
        # [wid_coord, hei_coord]
        rotated_point_coords = np.matmul(matrix, point_hom_coords.transpose()).transpose()

        if interpolation == 'bilinear':
            # keep the floating-point
            pass
        elif interpolation == 'nearest':
            rotated_point_coords = np.round(rotated_point_coords).astype(points_cp.dtype)

        # check
        out_of_range_flag_wid = (rotated_point_coords[:, 0] < 0) | (rotated_point_coords[:, 0] >= w)
        out_of_range_flag_hei = (rotated_point_coords[:, 1] < 0) | (rotated_point_coords[:, 1] >= h)
        out_of_range_flag = out_of_range_flag_wid | out_of_range_flag_hei

        # [[N, ], [N, 2]] -> [N, 3]
        rotated_points_cp = np.concatenate([points_cam_coord[:, None], rotated_point_coords], axis=1)

        # set unvalid
        rotated_points_cp[out_of_range_flag, :] = border_value

        return rotated_points_cp



    def __call__(self, results):
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """
        img = results["image"]
        points_cp = results.get("points_cp", None)
        img_label = results.get("image_label", None)
        img_has_annotations = img_label is not None


        degree = np.random.uniform(min(*self.degree), max(*self.degree))
        rot_valid = False

        if not img_has_annotations:
            return results
        else:
            seg_rot_temp = self.rotate_image(
                deepcopy(img_label),
                angle=degree,
                border_value=self.seg_pad_val,
                center=self.center,
                auto_bound=self.auto_bound,
                interpolation='nearest'
            )
            crop_bbox = self.get_crop_bbox(seg_rot_temp)
            # crop_valid = False
            for _ in range(self.try_times):
                seg_crop_temp = self.crop(deepcopy(seg_rot_temp), crop_bbox)
                # check the label without padded value
                labels, cnt = np.unique(seg_crop_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                label_valid = len(cnt) > 1 and (np.max(cnt) / np.sum(cnt)) < self.cat_max_ratio and (self.seg_pad_val not in labels)
                
                # check the kept points_cp
                points_cp_temp = self.crop_points_cp(deepcopy(points_cp), crop_bbox)
                num_points = points_cp_temp.shape[0]
                num_points_inside = (points_cp_temp[:, 0] > 0).sum()
                points_cp_valid = num_points_inside / num_points >= self.kept_min_ratio

                if label_valid & points_cp_valid:
                    # crop_valid = True
                    rot_valid = True
                    valid_crop_bbox = crop_bbox
                    valid_rot_degree = degree
                    break
                
                
                crop_bbox = self.get_crop_bbox(seg_rot_temp)
            


        if not rot_valid:
            return results
        else:
            # rot + crop for img
            # r_matrix: [2, 3]
            roted_img_temp = self.rotate_image(
                img,
                angle=valid_rot_degree,
                border_value=self.pal_val,
                center=self.center,
                auto_bound=self.auto_bound,
            )
            roted_img = self.crop(roted_img_temp, valid_crop_bbox)
            
            # rot + crop for img_label
            roted_img_label_temp = self.rotate_image(
                img_label,
                angle=valid_rot_degree,
                border_value=self.seg_pad_val,
                center=self.center,
                auto_bound=self.auto_bound,
                interpolation='nearest'
            )
            roted_img_label = self.crop(roted_img_label_temp, valid_crop_bbox)

            # rot + crop for points_cp
            # if points_cp.dtype is float, interpolation should be 'bilinear'
            # if points_cp.dtype is int, interpolation should be 'nearest'
            roted_points_cp_temp = self.rotate_points_cp(
                img,
                points_cp=points_cp,
                angle=valid_rot_degree,
                border_value=self.unvalid_cam_id,
                center=self.center,
                auto_bound=self.auto_bound,
                interpolation='nearest'
            )
            roted_points_cp = self.crop_points_cp(roted_points_cp_temp, valid_crop_bbox)


            # update the results
            results['rotate_valid'] = rot_valid
            results['image'] = roted_img
            results['cropped_shape'] = roted_img.shape
            results['points_cp'] = roted_points_cp
            results['image_label'] = roted_img_label

        return results



