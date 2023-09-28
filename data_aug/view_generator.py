import numpy as np

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]
#
#
# class CalandraViewGenerator(object):
#     """Take two random crops of one image as the query and key, considering two tactile images."""
#
#     def __init__(self, base_transform, n_views=2):
#         self.base_transform = base_transform
#         self.n_views = n_views
#
#     def __call__(self, x, gelA_image, gelB_image):
#         transformed_x = [self.base_transform(x) for i in range(self.n_views)]
#         transformed_gelA = [self.base_transform(gelA_image) for i in range(self.n_views)]
#         transformed_gelB = [self.base_transform(gelB_image) for i in range(self.n_views)]
#
#         return transformed_x, transformed_gelA, transformed_gelB
#
#
# class TouchAndGoViewGenerator(object):
#     """Take two random crops of one image as the query and key, considering one tactile image."""
#
#     def __init__(self, base_transform, n_views=2):
#         self.base_transform = base_transform
#         self.n_views = n_views
#
#     def __call__(self, x, gelA_image):
#         transformed_rgb = [self.base_transform(x) for i in range(self.n_views)]
#         transformed_tactile = [self.base_transform(gelA_image) for i in range(self.n_views)]
#         return transformed_rgb, transformed_tactile
