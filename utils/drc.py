import torch
import torch.nn as nn
import torchattacks
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from copy import deepcopy

class adv_dataset(Dataset):
    def __init__(self):
        self.images = None
        self.labels = None

    def append_data(self, images, labels):
        if self.images is None:
            self.images = images
            self.labels = labels
        else:
            self.images = torch.cat((self.images, images), dim=0)
            self.labels = torch.cat((self.labels, labels), dim=0)

    def __getitem__(self, item):
        img = self.images[item]
        label = self.labels[item]
        return img, label

    def __len__(self):
        return self.images.shape[0]

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean).to("cuda"))
        self.register_buffer('std', torch.Tensor(std).to("cuda"))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

def generate_adv_dataset(model):
    adv_train_dataset = adv_dataset()
    model = model.eval()
    atk_model = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 225, steps=10, random_start=True)
    # atk_model = torchattacks.FGSM(model, eps=8 / 255)
    transform_ = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

    train_dataset = datasets.STL10('../Dataset/data/stl10', split="train", download=True, transform=transform_)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8)

    for images, labels in trainloader:
        images = images.cuda()
        labels = labels.cuda()
        adv_images = atk_model(images, labels)
        adv_train_dataset.append_data(adv_images, labels)

    return adv_train_dataset

def layer_robustness_contribution(model, epsilon=0.1):
    criterion = nn.CrossEntropyLoss()
    trainloader = torch.utils.data.DataLoader(generate_adv_dataset(deepcopy(model)), batch_size=512, shuffle=True, num_workers=0)
    origin_total = 0
    origin_loss = 0.0
    origin_acc = 0
    with torch.no_grad():
        model.eval()
        for inputs, targets in trainloader:
            outputs = model(inputs)
            origin_total += targets.shape[0]
            origin_loss += criterion(outputs, targets).item() * targets.shape[0]
            _, predicted = outputs.max(1)
            origin_acc += predicted.eq(targets).sum().item()

        origin_acc /= origin_total
        origin_loss /= origin_total

    print("{:35}, Robust Loss: {:10.2f}, Robust Acc: {:10.2f}".format("Origin", origin_loss, origin_acc *100))

    model.eval()
    layer_robustness_contribution_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if "sub" in name:
                continue
            layer_robustness_contribution_dict[name] = 1e10

    for layer_name, _ in model.named_parameters():
        if "weight" in layer_name and layer_name[:-len(".weight")] in layer_robustness_contribution_dict.keys():
            cloned_model = deepcopy(model)
            for name, param in cloned_model.named_parameters():
                if name == layer_name:
                    param.requires_grad = True
                    init_param = param.detach().clone()
                else:
                    param.requires_grad = False
            optimizer = torch.optim.SGD(cloned_model.parameters(), lr=1)

            max_loss = 0.0
            min_acc = 0
            for epoch in range(10):
                for inputs, targets in trainloader:
                    optimizer.zero_grad()
                    outputs = cloned_model(inputs)
                    loss = -1 * criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                sd = cloned_model.state_dict()
                diff = sd[layer_name] - init_param
                times = torch.linalg.norm(diff ) /torch.linalg.norm(init_param)
                if times > epsilon:
                    diff = diff / times * epsilon
                    sd[layer_name] = deepcopy(init_param + diff)
                    cloned_model.load_state_dict(sd)
                with torch.no_grad():
                    total = 0
                    total_loss = 0.0
                    correct = 0
                    for inputs, targets in trainloader:
                        outputs = cloned_model(inputs)
                        total += targets.shape[0]
                        total_loss += criterion(outputs, targets).item() * targets.shape[0]
                        _, predicted = outputs.max(1)
                        correct += predicted.eq(targets).sum().item()
                    total_loss /= total
                    correct /= total

                if total_loss > max_loss:
                    max_loss = total_loss
                    min_acc = correct

            layer_robustness_contribution_dict[layer_name[:-len(".weight")]] = max_loss - origin_loss
            print("Epoch {:3}, {:35}, Layer Robustness Contribution : {:10.2f}, Dropped Robust Acc: {:10.2f}".format(
                epoch, layer_name[:-len(".weight")], max_loss - origin_loss, (origin_acc - min_acc) * 100))

    sorted_layer_robustness_contribution = sorted(layer_robustness_contribution_dict.items(), key=lambda x :x[1])
    for (k, v) in sorted_layer_robustness_contribution:
        print("{:35}, Robust Loss: {:10.2f}".format(k, v))

    total_items = len(sorted_layer_robustness_contribution)
    top_10_percent = int(total_items * 0.2)
    print("total_items:", total_items, "top_10_percent:", top_10_percent)
    # Extract the top 10% items based on values
    top_10_percent_items = sorted_layer_robustness_contribution[:top_10_percent]
    # Extract keys from the top 10% items
    top_10_percent_keys = [item[0] for item in top_10_percent_items]

    return sorted_layer_robustness_contribution, top_10_percent_keys


