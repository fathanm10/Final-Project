# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import torch.optim as optim
from torch.optim import lr_scheduler
import losses
from net.resnet import *
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(model_name,
             embedding_size=512,
             pretrained=True):
    if model_name == 'resnet50':
        return Resnet50(embedding_size=embedding_size, pretrained=pretrained)


def get_loss_func(loss_func_name,
                  num_classes,
                  embedding_size=512,
                  margin=0.1,
                  alpha=32):
    if loss_func_name == 'proxy_anchor':
        return losses.Proxy_Anchor(nb_classes=num_classes, sz_embed=embedding_size, mrg=margin, alpha=alpha)


def train_model(model_name,
                loss_func_name,
                num_classes,
                epochs,
                dataloader,
                use_loss_optimizer=False,
                embedding_size=512,
                pretrained=True,
                learning_rate=0.001,
                loss_learning_rate=0.01,
                margin=0.1,
                alpha=32,
                step_size=5,
                gamma=0.1,
                save_path=None):
    model = get_model(model_name, embedding_size, pretrained).to(device)
    loss_func = get_loss_func(loss_func_name, num_classes, embedding_size=embedding_size, margin=margin, alpha=alpha)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if use_loss_optimizer:
        loss_optimizer = optim.SGD(loss_func.parameters(), lr=loss_learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    then=time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_loss_optimizer:
                loss_optimizer.step()

            running_loss += loss.item()
            if (i+1)%int(len(dataloader)/5)==0 or i+1==len(dataloader):
                print(f'Step: [{i+1}/{len(dataloader)}] Epoch [{epoch+1}/{epochs}] Loss: {running_loss / len(dataloader):.4f} Time: {time.time() - then:.4f}')
        print(f'Epoch [{epoch+1}/{epochs}] Loss: {running_loss / len(dataloader):.4f} Time: {time.time() - then:.4f}')
        scheduler.step()

    print(f'Finished Training, Time: {time.time()-then:.4f}')
    if save_path != None:
        torch.save(model, save_path)
    return model
