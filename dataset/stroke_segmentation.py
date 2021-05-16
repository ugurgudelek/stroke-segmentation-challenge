# -*- encoding: utf-8 -*-
# @File    :   stroke_segmentation.py
# @Time    :   2021/05/16 01:32:17
# @Author  :   Ugur Gudelek
# @Contact :   ugurgudelek@gmail.com
# @Desc    :   None


class StrokeSegmentationDataset:
    def __init__(self):

        self.trainset = StrokeTorch(images=training_images,
                                    labels=training_labels)
        self.testset = StrokeTorch(images=test_images, labels=test_labels)


class StrokeSegmentationTorch(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
