import albumentations as A
import random

def augment_data(image, mask):
    #Geometric transformation
    geometric_transforms = []
    if random.random() > 0.45:
        geometric_transforms.append(A.HorizontalFlip(p=1.0))

    if random.random() > 0.6:
        geometric_transforms.append(A.Rotate(limit=20, p=1.0))

    if geometric_transforms:
        geo_transform = A.Compose(geometric_transforms)
        geo_result = geo_transform(image=image, mask=mask)

        image = geo_result['image']
        mask = geo_result['mask']

    #ShiftScaleRotate
    if random.random() > 0.6:
        shift_transform = A.ShiftScaleRotate(
            shift_limit=0.1,scale_limit=0.15,rotate_limit=15,p=1.0,border_mode=0)
        ssr_result = shift_transform(image=image, mask=mask)
        image = ssr_result['image']
        mask = ssr_result['mask']


    #Color transformation
    color_choice = random.choice(['brightness_contrast','hue_saturation','rgb_shift','none'])

    if color_choice == 'brightness_contrast':
        color_transform = A.RandomBrightnessContrast(
            brightness_limit=0.2,contrast_limit=0.2,p=1.0)

    elif color_choice == 'hue_saturation':
        color_transform = A.HueSaturationValue(
            hue_shift_limit=10,sat_shift_limit=15,val_shift_limit=10,p=1.0)

    elif color_choice == 'rgb_shift':
        color_transform = A.RGBShift(
            r_shift_limit=15,g_shift_limit=15,b_shift_limit=15,p=1.0)
    else:
        color_transform = None

    if color_transform:
        color_result = color_transform(image=image)
        image = color_result['image']

    return image, mask
