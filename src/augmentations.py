import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size=800):
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.1),
            A.MotionBlur(blur_limit=7, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3
        ),
    )


def get_val_transforms(img_size=800):
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3
        ),
    )
