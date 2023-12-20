import os
import shutil
import random
import torch
import torchvision
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

torch.manual_seed(0)

class ParkinsonDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs,transform):
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('png')]
            print(f'Found {len(images)}{class_name}')
            return images
        self.images={}
        self.class_names=['spiral-healthy','spiral-parkinson','wave-healthy','wave-parkinson' ]
        for c in self.class_names:
            self.images[c]=get_images(c)
        self.image_dirs=image_dirs
        self.transform=transform
    def __len__(self):
        return sum([len(self.images[c]) for c in self.class_names])
    def __getitem__(self, index):
        class_name=random.choice(self.class_names)
        index=index%len(self.images[class_name])
        image_name=self.images[class_name][index]
        image_path =os.path.join(self.image_dirs[class_name], image_name)
        image=Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)
        
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224,224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
])
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])
])

train_dirs = {
    'spiral-healthy': 'dataset/train/spiral-healthy',
    'spiral-parkinson': 'dataset/train/spiral-parkinson',
    'wave-healthy': 'dataset/train/wave-healthy',
    'wave-parkinson': 'dataset/train/wave-parkinson'
    
}
train_dataset=ParkinsonDataset(train_dirs, train_transform)

test_dirs = {
    'spiral-healthy': 'dataset/test/spiral-healthy',
    'spiral-parkinson': 'dataset/test/spiral-parkinson',
    'wave-healthy': 'dataset/test/wave-healthy',
    'wave-parkinson': 'dataset/test/wave-parkinson'
}
test_dataset = ParkinsonDataset(test_dirs, test_transform)


batch_size=6
dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print('Num of training batches', len(dl_train))
print('Num of test batches', len(dl_test))

class_names=train_dataset.class_names
def show_images(images, labels, preds):
    plt.figure(figsize=(8,4))
    for i, image in enumerate(images):
        plt.subplot(1,6,i+1, xticks=[], yticks=[])
        image=image.numpy().transpose((1,2,0))
        mean=np.array([0.485,0.456,0.406])
        std= np.array([0.229, 0.224, 0.225])
        image=image*std/mean
        image=np.clip(image,0.,1.)
        plt.imshow(image)
        col = 'green' if preds[i]==labels[i] else 'red'
        plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color=col)
    plt.tight_layout()
    plt.show()
    
images, labels =next(iter(dl_train))
show_images(images, labels, labels)

images, labels =next(iter(dl_test))
show_images(images, labels, labels)

resnet18 =torchvision.models.resnet18(pretrained=True)
print(resnet18)

resnet18.fc=torch.nn.Linear(in_features=512, out_features=4)
loss_fn=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(resnet18.parameters(), lr=3e-5)
def show_preds():
    resnet18.eval()
    images, labels =next(iter(dl_test))
    outputs = resnet18(images)
    _, preds=torch.max(outputs, 1)
    show_images(images, labels, preds)
show_preds()

def train(epochs):
    print('Starting training..')
    for e in range(0, epochs):
        print(f'Starting epoch {e+1}/{epochs}')
        print('='*20)
        train_loss=0
        resnet18.train()
        for train_step, (images, labels) in enumerate(dl_train):
            optimizer.zero_grad()
            outputs=resnet18(images)
            loss=loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss=loss.item()
            if train_step%20==0:
                print('Evaluating at step', train_step)
                acc=0.
                val_loss=0.
                resnet18.eval()
                for val_step,(images, labels) in enumerate(dl_test):
                    outputs=resnet18(images)
                    loss=loss_fn(outputs, labels)
                    val_loss+=loss.item()
                    _,preds=torch.max(outputs, 1)
                    acc+=sum(preds==labels).numpy()
                val_loss/=(val_step+1)
                acc=acc/len(test_dataset)
                print(f'Val loss: {val_loss:.4f}, Acc: {acc:.4f}')
                show_preds()
                resnet18.train()
                if acc >0.95:
                    print('Performance condition satisfied..')
                    return
        train_loss/=(train_step+1)
        print(f'Training loss: {train_loss:.4f}')
train(epochs=10)
show_preds()