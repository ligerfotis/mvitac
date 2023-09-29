from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from generate_dataset import TouchFolderLabel, CalandraLabel


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    # @staticmethod
    # def get_simclr_pipeline_transform(size, s=1):
    #     """Return a set of data augmentation transformations as described in the SimCLR paper."""
    #     color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    #     data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
    #                                           transforms.RandomHorizontalFlip(),
    #                                           transforms.RandomApply([color_jitter], p=0.8),
    #                                           transforms.RandomGrayscale(p=0.2),
    #                                           GaussianBlur(kernel_size=int(0.1 * size)),
    #                                           transforms.ToTensor()])
    #     return data_transforms

    def get_dafault_transform(self, size):
        data_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                              transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                                              transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.50),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225]),
                                              # transforms.RandomGrayscale(p=0.2),
                                              ])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'tag': lambda: TouchFolderLabel(self.root_folder,
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_dafault_transform(224), n_views),
                                                          mode='pretrain'),
                          'calandra_label': lambda: CalandraLabel(self.root_folder,
                                                                  transform=ContrastiveLearningViewGenerator(
                                                                      self.get_dafault_transform(224),
                                                                      n_views), mode='test')
                          }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise NotImplementedError

        return dataset_fn()
