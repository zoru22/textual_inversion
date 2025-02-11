import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random

imagenet_templates_smallest = [
    'a digital illustration of a {}',
]

imagenet_templates_small = [
    'a digital illustration of a {}',
    'a rendering of a {} pokemon',
    'a cropped digital painting of the {}',
    'the digital illustration of a {}',
    'a digital illustration of a clean {}',
    'a digital illustration of a dirty {}',
    'a dark digital illustration of the {}',
    'a digital illustration of my {}',
    'a digital illustration of the cool {}',
    'a close-up digital illustration of a {}',
    'a bright digital illustration of the {}',
    'a cropped digital illustration of a {}',
    'a digital illustration of the {}',
    'a good digital illustration of the {}',
    'a digital illustration of one {}',
    'a close-up digital illustration of the {}',
    'a rendition of the {} pokemon',
    'a digital illustration of the clean {} pokemon',
    'a rendition of a {}',
    'a digital illustration of a nice {} pokemon',
    'a good digital illustration of a {}',
    'a digital illustration of the nice {}',
    'a digital illustration of the small {}',
    'a digital illustration of the weird {}',
    'a digital illustration of the large {} pokemon',
    'a digital illustration of a cool {}',
    'a digital illustration of a small {}',
]

imagenet_dual_templates_small = [
    'a digital illustration of a {} with {}',
    'a rendering of a {} with {}',
    'a cropped digital illustration of the {} with {}',
    'the digital illustration of a {} with {}',
    'a digital illustration of a clean {} with {}',
    'a digital illustration of a dirty {} with {}',
    'a dark digital illustration of the {} with {}',
    'a digital illustration of my {} with {}',
    'a digital illustration of the cool {} with {}',
    'a close-up digital illustration of a {} with {}',
    'a bright digital illustration of the {} with {}',
    'a cropped digital illustration of a {} with {}',
    'a digital illustration of the {} with {}',
    'a good digital illustration of the {} with {}',
    'a digital illustration of one {} with {}',
    'a close-up digital illustration of the {} with {}',
    'a rendition of the {} with {}',
    'a digital illustration of the clean {} with {}',
    'a rendition of a {} with {}',
    'a digital illustration of a nice {} with {}',
    'a good digital illustration of a {} with {}',
    'a digital illustration of the nice {} with {}',
    'a digital illustration of the small {} with {}',
    'a digital illustration of the weird {} with {}',
    'a digital illustration of the large {} with {}',
    'a digital illustration of a cool {} with {}',
    'a digital illustration of a small {} with {}',
]

multi_templates_small = [
    'digital sketch person {} of {}',
    'rendering of {} of a {}',
    'cropped {} art of a {}',
    'cropped {} digital art {}',
    'digital {} art of {}',
    'digital {} painting of {}',
    '{} illustration of a dirty {}',
    'digital {} art of {}',
    'middle {} of {}',
    'mid {} transformation of {}',
    '{} of a {}',
    'closeup digital {} art of {}',
    'good digital {} sketch of {}',
    'cropped {} digital art {}',
    'digital drawing of mid {} of {}',
    'great art of {} partial, {}',
    '{} sketch of {}',
    'sketchy {} art, {}',
    'hd 8k {} image of a character, {}',
    '{} pixiv high-quality, {}',
    '4k {} art {} pretty',
    '{} rendition of {}',
    'hd {} art of a nice {}',
    'high-quality digital {} painting, {}',
    '{} painting of a surprised {}',
    '{} stoic {} with plain background',
    'happy {} of a transforming {}',
]

imagenet_multi_templates_small = [
    'a painting in the style of {} with {}',
    'a rendering in the style of {} with {}',
    'a cropped painting in the style of {} with {}',
    'the painting in the style of {} with {}',
    'a clean painting in the style of {} with {}',
    'a dirty painting in the style of {} with {}',
    'a dark painting in the style of {} with {}',
    'a cool painting in the style of {} with {}',
    'a close-up painting in the style of {} with {}',
    'a bright painting in the style of {} with {}',
    'a cropped painting in the style of {} with {}',
    'a good painting in the style of {} with {}',
    'a painting of one {} in the style of {}',
    'a nice painting in the style of {} with {}',
    'a small painting in the style of {} with {}',
    'a weird painting in the style of {} with {}',
    'a large painting in the style of {} with {}',
]


ran_once = False
def validate_personalized_strs():
    global ran_once

    if ran_once:
        return
    for x in multi_templates_small:
        assert x.count("{}") == 2, "multi_templates_small. string: [" + x + "] requires exactly two {}"
    ran_once=True

validate_personalized_strs()
per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

precomputed_prompts = list()

class PersonalizedBase(Dataset):
    # TODO: Should we pass in the seed for slightly-more-reproducible training runs?
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 mixing_prob=0.25,
                 coarse_class_text=None,
                 ):
        global precomputed_prompts
        if per_image_tokens:
            raise Exception('per image tokens should not be used. this feature is probably completely broken now')

        self.data_root = data_root

        self.image_paths = list()

        # One collection of precomputed prompts per Image.
        self.precomputed_prompts = list()
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob

        placeholder_string = placeholder_token

        self.coarse_class_text = coarse_class_text
        if self.coarse_class_text:
            # placeholder_string = f"{self.coarse_class_text} {placeholder_token}"
            raise Exception('coarse class text isnt supported ATM')

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

        images_in = os.listdir(self.data_root)

        compute_prompts = len(precomputed_prompts) == 0
        for file_path in images_in:
            self.image_paths.append(os.path.join(self.data_root, file_path))
            descriptive, _ = os.path.splitext(file_path)
            splitted = descriptive.split('-')
            assert len(splitted) > 1, 'image names must be of form: ####-description-here'
            # This breaks non-descriptor image setups
            assert len(splitted[0]) == 4, f'image name: {descriptive} must be of the form ####-description-here'

            descriptive = ' '.join(splitted[1:])

            if compute_prompts:
                precomputed_prompts.append([x.format(placeholder_string,descriptive) for x in imagenet_multi_templates_small])

        self._length = self.num_images = len(self.image_paths)

        if len(precomputed_prompts) != self._length:
            raise Exception(f'Expected # of image prompts: {self._length} does not match actual: {len(precomputed_prompts)}')

        if set == "train":
            self._length = self.num_images * repeats

        self.flip = transforms.RandomHorizontalFlip(p=flip_p)



    def __len__(self):
        return self._length

    def __getitem__(self, i):
        global precomputed_prompts
        selected_idx = i % self.num_images

        current_img = self.image_paths[selected_idx]

        # TODO: what if we preloaded the images into memory?
        image = Image.open(current_img)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        prompt_list = precomputed_prompts[selected_idx]
        text = random.choice(prompt_list)

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)

        example = {
            "caption": text,
            "image": (image / 127.5 - 1.0).astype(np.float32),
        }
        return example