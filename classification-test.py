from mimetypes import init
import warnings

warnings.filterwarnings("ignore", module="matplotlib\..*")

from pathlib import Path
from cv2 import log
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

import pandas as pd

from berries.experiments.experiment import Experiment
from berries.metric import metrics
from berries.logger import MultiLogger
from berries.trainer.base import BaseTrainer

from dataset.stroke import StrokeClassificationDataset, StrokeClassificationTorch
from model.cnn import VGG16, CNN, DenseNet, ResNet, CustomCNN

from metric import classification as local_metrics
from dataset.stroke import Stroke

PNG_PATH = Path('input/teknofest/contest/input/classification/PNG_OTURUM1')
# PNG_PATH = Path('input/teknofest/raw/KANAMA/PNG')
PRETRAINED_MODEL_PATH = Path(
    './checkpoints/classification/170/model-optim.pth')
OUTPUT_PATH = Path('input/teknofest/contest/output/classification')
HOLD_PROBABILITY = 1.
INFER_LABEL = False

params = {
    'resume': False,
    'pretrained': True,
    'pretrained_path': PRETRAINED_MODEL_PATH,
    'device': 'cuda'
}

alpha = 0.333
# Create netcdf4 file for faster reading
# Stroke(root=Path('./input/stroke'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNet(net_type='ResNet-50',
               pre_trained=False,
               req_grad=True,
               out_channels=2,
               input_dim=(3, 512, 512))
print(model)

# transform = transforms.Compose([
#     transforms.RandomApply([
#         transforms.RandomRotation((-180, 180)),
#         transforms.ColorJitter(),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomVerticalFlip(p=0.5),
#         transforms.GaussianBlur(kernel_size=51)
#     ],
#                            p=0.5)
# ])

xr_dataset = Stroke.from_path_classification(png_path=PNG_PATH,
                                             infer_label=INFER_LABEL,
                                             hold_prop=HOLD_PROBABILITY)
torch_dataset = StrokeClassificationTorch(xr_dataset.image.values,
                                          xr_dataset.label.values,
                                          transform=None)

trainer = BaseTrainer(model=model,
                      criterion=torch.nn.CrossEntropyLoss(
                          weight=torch.tensor([alpha, 1 - alpha]).to(device)),
                      metrics=[
                          local_metrics.Accuracy(),
                          local_metrics.MeanMetric(),
                          local_metrics.Recall(),
                          local_metrics.Specificity()
                      ],
                      hyperparams=dict(),
                      params=params,
                      optimizer=None,
                      scheduler=None,
                      logger=None)

OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

predictions, targets = trainer.transform(dataset=torch_dataset,
                                         classification=True)
prediction_dataframe = pd.DataFrame({
    'ID': xr_dataset.id.values,
    'ETiKET': predictions.squeeze(),
    # 'target': targets.squeeze()
})# yapf:disable

prediction_dataframe.to_csv(OUTPUT_PATH / 'OTURUM1.csv', index=False, sep=';')

print(prediction_dataframe)

score, _ = trainer.score(dataset=torch_dataset,
                         batch_size=16,
                         classification=False)
print(score)
