from torch.utils.data import DataLoader
from dataset.dataset import INaturalistDataset
from dataset.transforms import get_transforms
# from training.trainer import train
import torch

iNat_root = "/speech/sarath/Database/inaturalist_12K"
train_dataset = INaturalistDataset(
    root_dir=iNat_root,
    split="train",
    transform=get_transforms(train=True, augment=True)
)

val_dataset = INaturalistDataset(
    root_dir=iNat_root,
    split="val",
    transform=get_transforms(train=False)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

images, labels = next(iter(train_loader))
print(images.shape, labels.shape)

criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# train(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     criterion=criterion,
#     optimizer=optimizer,
#     device="cuda",
#     epochs=10
# )