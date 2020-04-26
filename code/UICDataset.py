class UICDataset:
    """
    Custom Datset class to use Dataloader
    """
    def __init__(self, reviews, other_data, target):
        self.reviews = reviews
        self.other_data = other_data
        self.target = target

    def __getitem__(self, index):
        x = self.reviews[index]
        y = self.target[index]
        z = self.other_data[index]

        return [x, y, z]

    def __len__(self):
        return len(self.reviews)