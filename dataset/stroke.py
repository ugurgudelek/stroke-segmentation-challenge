# -*- encoding: utf-8 -*-
# @File    :   stroke-experiment.py
# @Time    :   2021/05/15 17:45:06
# @Author  :   Ugur Gudelek, Gorkem Can Ates
# @Contact :   ugurgudelek@gmail.com
# @Desc    :   None


class StrokeClassification:
    def __init__(self):
        self.trainset = StrokeTorch(images=training_mage)
        self.testset = StrokeTorch()


class StrokeClassificationTorch(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


class StrokeSegmentation:
    def __init__(self):
        self.trainset = StrokeTorch(images=training_mage)
        self.testset = StrokeTorch()


class StrokeSegmentationTorch(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass