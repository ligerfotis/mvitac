# Import necessary libraries for image transformations and datasets
from torchvision.transforms import transforms
from torchvision import transforms
from data_aug.view_generator import ContrastiveLearningViewGenerator
from generate_dataset import TouchFolderLabel, CalandraLabel

# Define the Contrastive Learning Dataset class
class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        # Initialize the root folder for the dataset
        self.root_folder = root_folder

    def get_dafault_transform(self, size):
        # Define the default data transformations for training
        # Includes resizing, random crops, horizontal flip, grayscale conversion, normalization, etc.
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

    def get_test_transform(self, size):
        # Define the data transformations for testing
        # Only includes resizing and normalization
        data_transforms = transforms.Compose([transforms.Resize((256, 256)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225]),
                                              ])
        return data_transforms
    def get_dataset(self, name, n_views):
        # Define valid datasets and their corresponding transformations
        # Use lambdas to define how each dataset should be loaded and transformed
        valid_datasets = {'tag_train': lambda: TouchFolderLabel(self.root_folder,
                                                                transform=ContrastiveLearningViewGenerator(
                                                                    self.get_dafault_transform(224), n_views),
                                                                mode='pretrain'),
                          'tag_test': lambda: TouchFolderLabel(self.root_folder,
                                                               transform=ContrastiveLearningViewGenerator(
                                                                   self.get_test_transform(224), n_views),
                                                               mode='test'),

                          'calandra_label_train': lambda: CalandraLabel(self.root_folder,
                                                                        transform=ContrastiveLearningViewGenerator(
                                                                            self.get_dafault_transform(224),
                                                                            n_views), mode='train'),
                          'calandra_label_test': lambda: CalandraLabel(self.root_folder,
                                                                       transform=ContrastiveLearningViewGenerator(
                                                                           self.get_test_transform(224),
                                                                           n_views), mode='test')
                          }

        # Try to get the dataset function based on the provided name
        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            # Raise error if the dataset name is not recognized
            raise NotImplementedError

        # Return the dataset by calling the function
        return dataset_fn()
