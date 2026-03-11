from torch.utils.data import DataLoader
from dataset.dataset import INaturalistDataset
from dataset.transforms import get_transforms
# from training.trainer import train
import torch
from models.imageCNN import imageCNN

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


# model = imageCNN(num_classes=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_channels=3
model = imageCNN(input_channels=3,num_classes=10).to(device)

x = torch.randn(1,3,224,224)
x = x.to(device)
print(model.parameters())

for name, param in model.named_parameters():
    print(name,param.size())

y = model(x)

print(y.shape)
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