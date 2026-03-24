# unet-pet-segmentation
## Model Performance

Developed a semantic segmentation model based on the **U-Net** architecture using the **Oxford-IIIT Pet Dataset**. Implemented preprocessing steps including image resizing, normalization, and mask encoding, and trained the model to segment pets from background.

Achieved strong performance on the test set:

| Metric    | Score  |
|-----------|--------|
| Dice Score | 0.9026 |
| IoU        | 0.8245 |
| Precision  | 0.8906 |
| Recall     | 0.9172 |

The model was trained for **100 epochs**, showing consistent improvement and stable convergence after **~epoch 80**, with closely aligned training and validation curves indicating good generalization and minimal overfitting.

Training plots, data augmentation samples, and segmentation results (ground truth vs. predicted masks) are available in the [`images/`](images/) directory.
