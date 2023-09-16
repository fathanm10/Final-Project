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
from net.pfe import PFE
import time
import pytorch_metric_learning as pml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(model_name,
             embedding_size=512,
             pretrained=True):
    if model_name == 'resnet50':
        return Resnet50(embedding_size=embedding_size, pretrained=pretrained)
    elif model_name == 'pfe':
        backbone = Resnet50(embedding_size=embedding_size, pretrained=pretrained, freeze_all=True, conv_final=True)
        return PFE(embedding_size, backbone)


def get_loss_func(loss_func_name,
                  num_classes=None,
                  embedding_size=512,
                  margin=0.1,
                  alpha=32
                 ):
    if loss_func_name == 'proxy_anchor':
        return losses.Proxy_Anchor(nb_classes=num_classes, sz_embed=embedding_size, mrg=margin, alpha=alpha)
    if loss_func_name == 'proxy_anchor_lib':
        return pml.losses.ProxyAnchorLoss(num_classes, embedding_size, margin=margin, alpha=alpha).to(device)
    if loss_func_name == 'mutual_likelihood_score':
        return losses.MutualLikelihoodScoreLoss()


def get_optimizer(optimizer, param, learning_rate, momentum, weight_decay):
    if optimizer=='adam':
        return optim.Adam(param, lr=learning_rate)
    elif optimizer=='sgd':
        return optim.SGD(param, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


def train_model(model_name,
                loss_func_name,
                num_classes,
                epochs,
                dataloader,
                optimizer='adam',
                use_loss_optimizer=False,
                embedding_size=512,
                pretrained=True,
                learning_rate=0.001,
                loss_learning_rate=0.01,
                margin=0.1,
                alpha=32,
                step_size=5,
                gamma=0.5,
                momentum=0.9,
                weight_decay=5e-4,
                save_path=None,
                verbose=2):
    model = get_model(model_name, embedding_size, pretrained).to(device)
    loss_func = get_loss_func(loss_func_name, num_classes, embedding_size=embedding_size, margin=margin, alpha=alpha)
    optimizer = get_optimizer(optimizer, model.parameters(), learning_rate, momentum, weight_decay)
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
            loss = loss_func(*outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_loss_optimizer:
                loss_optimizer.step()

            running_loss += loss.item()
            if verbose > 1 and (i+1==len(dataloader) or (i+1)%10==0):
                print(f'Step: [{i+1}/{len(dataloader)}] Epoch [{epoch+1}/{epochs}] Loss: {running_loss / len(dataloader):.4f} Time: {time.time() - then:.4f}')
        if verbose > 0:
            print(f'Epoch [{epoch+1}/{epochs}] Loss: {running_loss / len(dataloader):.4f} Time: {time.time() - then:.4f}')
        scheduler.step()

    print(f'Finished Training, Time: {time.time()-then:.4f}')
    if save_path != None:
        torch.save(model, save_path)
    return model