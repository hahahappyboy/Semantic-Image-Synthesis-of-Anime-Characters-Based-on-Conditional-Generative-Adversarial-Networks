import random
import torch
from torchvision import transforms as TR
import os
from PIL import Image

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        opt.load_size = 512
        opt.crop_size = 512
        opt.label_nc = 11
        opt.contain_dontcare_label = False
        opt.semantic_nc = 11 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.class_path = opt.class_dir
        with open(self.class_path, "r", encoding="utf-8") as f:  # 打开文件
            cls = f.readlines()  # 读取文件
        cls = cls[1:]
        self.class_dict = {}
        for c in cls:
            ls = c.split()
            self.class_dict[ls[0]] = ls[1]
        print(self.class_dict)

        self.opt = opt
        self.for_metrics = for_metrics

        self.images, self.labels, self.edges,self.paths = self.list_images()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx])
        edge = Image.open(self.edges[idx])
        # print(self.images[idx])
        image, label,edge = self.transforms(image, label, edge)

        label = label * 255
        cls = self.class_dict[self.images[idx].split('\\')[-1]]
        clss = torch.tensor(int(cls), device=torch.device("cuda:"+self.opt.gpu_ids if torch.cuda.is_available() else "cpu"))

        edge = edge*255
        return {"image": image, "label": label, "name": self.images[idx], "edge":edge,'class': clss}

    def list_images(self):
        mode = "test" if self.opt.phase == "test" or self.for_metrics else "train"
        images = []
        path_img = os.path.join(self.opt.dataroot,mode+'_img')
        for city_folder in sorted(os.listdir(path_img)):
            item = os.path.join(path_img, city_folder)
            # for item in sorted(os.listdir(cur_folder)):
            images.append(item)
        labels = []
        path_lab = os.path.join(self.opt.dataroot,mode+'_label')
        for city_folder in sorted(os.listdir(path_lab)):
            item = os.path.join(path_lab, city_folder)
            # for item in sorted(os.listdir(cur_folder)):
            labels.append(item)
        edges = []
        path_canny = os.path.join(self.opt.dataroot, mode + '_edge')
        for city_folder in sorted(os.listdir(path_canny)):
            item = os.path.join(path_canny, city_folder)
            # for item in sorted(os.listdir(cur_folder)):
            edges.append(item)

        assert len(images)  == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))
        # for i in range(len(images)):
        #     assert images[i].replace("_leftImg8bit.png", "") == labels[i].replace("_gtFine_labelIds.png", ""),\
        #         '%s and %s are not matching' % (images[i], labels[i])
        return images, labels, edges,(path_img, path_lab)
    def transforms(self, image, label,edge):
        # print('image.size='+str(image.size))
        # print('label.size='+str(label.size))
        # print()
        assert image.size == label.size
        # resize
        new_width, new_height = (int(self.opt.load_size / self.opt.aspect_ratio), self.opt.load_size)
        image = TR.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        label = TR.functional.resize(label, (new_width, new_height), Image.NEAREST)
        edge = TR.functional.resize(edge, (new_width, new_height), Image.NEAREST)
        # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
                edge = TR.functional.hflip(edge)
        # to tensor
        image = TR.functional.to_tensor(image)
        label = TR.functional.to_tensor(label)
        edge = TR.functional.to_tensor(edge)
        # normalize
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return image, label ,edge
