# stroke-segmentation-challenge
Repo for Teknofest Saglikta Yapay Zeka Challenge

 # Installation
  * Download cuda from [nvidia-cuda-11.1.1](https://developer.nvidia.com/cuda-11.1.1-download-archive)
 
 * Clone the **[berries](https://github.com/ugurgudelek/berries)** repository

 * Enter into the project **root** directory
 * `conda env create --file environment.yml`
 * `conda activate stroke`
 * `conda develop ../berries`


# Todos

### Step 0
 * Create development medium
   * Install berries
   * Run demo experiment to test eveything is OK 
  
### Step 1
 * Implement abstract dataset classes for both classificaton and segmentation
 * Implement basic model class e.g. CNN, VGG, Resnet
 * Implement loss function for both classificaton and segmentation
 * Train the basic model
 * Implement result analysis methods

### Step 2
 * Implement more advanced models
 * Tryout new ideas

# Evaluation metrics
Phase I evaluation metric = (Recall + False Positive Rate)/2

Phase I criterion: Phase I evaluation metric > 0.75
Phase II criterion: Intersection over Union

# Ideas
 * Beyond the pixel-wise loss for topology-aware delineation
 * Transfer learning strategies for both classificaton and segmentation


# External Dataset
 * Brain CT Images with Intracranial Hemorrhage Mask
 * cancer_classiff
 * CT Medical Images
 * Unet_pretrained_Brain MRI segmentation

