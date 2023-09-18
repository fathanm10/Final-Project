import numpy as np
import torch
import logging
import losses
import json
from tqdm import tqdm
import torch.nn.functional as F
import math
import umap.umap_
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from dataset.utils import get_num_classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def predict_batchwise(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()
    
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = model(J.cuda())
                    if type(J) is tuple:
                        J = J[0]

                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    return [torch.stack(A[i]) for i in range(len(A))]

def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, *_ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T==class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean

def evaluate_cos(model, dataloader):
    nb_classes = get_num_classes(dataloader.dataset)
    
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    xs = []
    
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y = Y.float().cpu()
    
    recall = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall

def evaluate_cos_Inshop(model, query_dataloader, gallery_dataloader):
    nb_classes = query_dataloader.dataset.nb_classes()
    
    # calculate embeddings with model and get targets
    query_X, query_T = predict_batchwise(model, query_dataloader)
    gallery_X, gallery_T = predict_batchwise(model, gallery_dataloader)
    
    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)
    
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 50
    Y = []
    xs = []
    
    cos_sim = F.linear(query_X, gallery_X)

    def recall_k(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0

        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]

            thresh = torch.max(pos_sim).item()

            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1
            
        return match_counter / m
    
    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 10, 20, 30, 40, 50]:
        r_at_k = recall_k(cos_sim, query_T, gallery_T, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
                
    return recall

def evaluate_cos_SOP(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)
    
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 1000
    Y = []
    xs = []
    for x in X:
        if len(xs)<10000:
            xs.append(x)
        else:
            xs.append(x)            
            xs = torch.stack(xs,dim=0)
            cos_sim = F.linear(xs,X)
            y = T[cos_sim.topk(1 + K)[1][:,1:]]
            Y.append(y.float().cpu())
            xs = []
            
    # Last Loop
    xs = torch.stack(xs,dim=0)
    cos_sim = F.linear(xs,X)
    y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y.append(y.float().cpu())
    Y = torch.cat(Y, dim=0)

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 10, 100, 1000]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    return recall


def evaluate_accuracy(model, dataloader):
    model.eval() 

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(report)


def pair_MLS_score(x1, x2, sigma_sq1=None, sigma_sq2=None):
    mu1, mu2 = np.array(x1), np.array(x2)
    sigma_sq1, sigma_sq2 = np.array(sigma_sq1), np.array(sigma_sq2)
    sigma_sq_mutual = sigma_sq1 + sigma_sq2
    dist = np.sum(np.square(mu1 - mu2) / sigma_sq_mutual + np.log(sigma_sq_mutual), axis=1)
    return -dist


def visualize_umap(model, dataloader, mode=0):
    def scatter(model, images, labels):
        logits = model(images)
        if type(logits) is tuple:
            logits = logits[0]
        reduced_embeddings = reducer.fit_transform(logits.cpu())
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='Spectral')
    
    model.to(device)
    all_images = []
    all_labels = []
    reducer = umap.umap_.UMAP()
    with torch.no_grad():
        model.eval()
        if mode==0:
            images, labels = next(iter(dataloader))
            images = images.to(device)
            scatter(model, images, labels)
        elif mode==1:
            for i, (images, labels) in enumerate(dataloader):
                print(f'Progress: {i+1}/{len(dataloader)} batch')
                images = images.to(device)
                scatter(model, images, labels)
        elif mode==2:
            for i, (images, labels) in enumerate(dataloader.dataset):
                images = images.to(device).unsqueeze(dim=0)
                all_images.append(images)
                all_labels.append(labels)
            all_images = torch.cat(all_images, dim=0)
            scatter(model, all_images, all_labels)
    plt.title("UMAP Visualization of Predicted Classes and Labels")
    plt.xlabel("umap1")
    plt.ylabel("umap2")
    plt.colorbar()
    plt.show()


def display_images(dataloader, h, w):
    i = 0
    stop = False
    for images, labels in dataloader:
        if stop:
            break
        for j in range(images.shape[0]):  # elements in batch
            if i == h*w:
                stop = True
                break
            plt.subplot(h,w,i+1)
            plt.imshow(images[j][0])
            i += 1
