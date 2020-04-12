import cv2
import os

from torch.utils.data import Dataset
from imgaug import augmenters as iaa
from torchvision import transforms as T


class MegaAgeAsianDatasets(Dataset):
    def __init__(self, image_txt_path, age_txt_path, base_path, augment=False, input_size=64):
        self.image_txt = image_txt_path
        self.age_txt = age_txt_path
        self.base_path = base_path
        self.augment = augment
        self.input_size = input_size
    
    def __len__(self):
        return len(self.image_txt)
    
    def __getitem__(self, index):
        
        image, _ = self.read_images(index)
        label = int(self.age_txt[index].strip('\n'))
            
        if self.augment:
            image = self.augmentor(image)
        
        image = self.preprocessing(image)

        return image.float(), label
    
    def read_images(self, index):
        filename = self.image_txt[index].strip('\n')
        image_path = os.path.join(self.base_path, filename)
        image = cv2.imread(image_path)
        return image, image_path
    
    def augmentor(self, image):
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.SomeOf((0, 4), [
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
            ]),
            iaa.OneOf([
                iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(2, 7)),  # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 11)),  # blur image using local medians with kernel sizes between 2 and 7
            ]),
            # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
        ], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug

    def preprocessing(self, image):
        image = T.Compose([
        T.ToPILImage(),
        T.Resize((self.input_size, self.input_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(image)
        
        return image


if __name__ == "__main__":
    pass
