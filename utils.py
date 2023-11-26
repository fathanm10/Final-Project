import numpy as np
import torch
import logging
import losses
import json
from tqdm.notebook import tqdm
import torch.nn.functional as F
import math
import umap.umap_
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, pairwise, roc_curve, auc
from dataset.utils import get_num_classes
import time
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# +
# Code taken from https://github.com/tjddus9597/Proxy-Anchor-CVPR2020/blob/master/code/utils.py
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
                    J = model(J.cuda().float())
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


# -

def evaluate_accuracy(model, dataloader):
    model.eval() 

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device).float()
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    report = classification_report(true_labels, predicted_labels)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    print(report)


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
            images = images.to(device).float()
            scatter(model, images, labels)
        elif mode==1:
            for i, (images, labels) in enumerate(dataloader):
                print(f'Progress: {i+1}/{len(dataloader)} batch')
                images = images.to(device).float()
                scatter(model, images, labels)
        elif mode==2:
            for i, (images, labels) in enumerate(dataloader.dataset):
                images = images.to(device).float().unsqueeze(dim=0)
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
    dataset=dataloader
    if hasattr(dataloader, 'dataset'):
        dataset = dataloader.dataset
    plt.figure(figsize=(3*w,3*h))
    for images, labels in dataset:
        if i == h*w:
            break
        plt.subplot(h,w,i+1)
        plt.imshow(np.transpose(images, axes=[1,2,0]))
        i += 1


def estimate_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Estimated model size: {:.3f}MB'.format(size_all_mb))


'''
@article{shi2019PFE,
  title = {Probabilistic Face Embeddings},
  author = {Shi, Yichun and Jain, Anil K.},
  booktitle = {arXiv:1904.09658},
  year = {2019}
}
'''
def mls_distance(x1, x2):
    mu1, sigma_sq1 = [np.array(i.cpu()) for i in x1]
    mu2, sigma_sq2 = [np.array(i.cpu()) for i in x2]
    sigma_sq_mutual = sigma_sq1 + sigma_sq2   # must be positive for np.log to work
    dist = np.sum(np.square(mu1 - mu2) / sigma_sq_mutual + np.log(sigma_sq_mutual), axis=1)
    return -dist


def similarity_score(x1, x2, metric):
    func = {
        'cosine' : pairwise.cosine_similarity,   # embedding
        'euclidean' : pairwise.euclidean_distances, # mu
        'mls' : mls_distance   # mu, log_sigma
    }
    
    if metric!='mls':
        if type(x1) is tuple:
            x1 = x1[0].cpu()
        else:
            x1 = x1.cpu()
        if type(x2) is tuple:
            x2 = x2[0].cpu()
        else:
            x2 = x2.cpu()
    return func[metric](x1, x2)


# +
def evaluate_pair(model, dataset, metric='cosine', threshold=None):
    model.to(device)
    model.eval()
    with torch.no_grad():
        sim_scores = []
        labels = []
        img1s = []
        img2s = []
        for img1, img2, label in dataset:
            img1s.append(img1.to(device).unsqueeze(dim=0))
            img2s.append(img2.to(device).unsqueeze(dim=0))
            labels.append(label)
        img1s = torch.cat(img1s, dim=0)
        img2s = torch.cat(img2s, dim=0)
        x1 = model(img1s.to(device)).cpu()
        x2 = model(img2s.to(device)).cpu()
        sim_score = similarity_score(x1,x2,metric)
        sim_scores.append(sim_score)
    sim_scores = np.array(sim_scores)
    labels = np.array(labels)
    
    # choose threshold with most accuracies
    label_vec = (sim_scores>0.5) == labels
    if threshold != None:
        label_vec = (sim_scores>threshold) == labels
        far, tar, thresholds = roc_curve(np.repeat(labels, sim_score.shape[0]), sim_score.reshape(-1))
        # Calculate AUC using scikit-learn
        auc_score = auc(far, tar)
        return far, tar, auc_score
    
#     score_pos = sim_scores[label_vec==True]
#     thresholds = np.sort(score_pos)
    thresholds = np.array([i/4 for i in range(21)])
    accuracies = np.zeros(np.size(thresholds))
    for i, threshold in enumerate(thresholds):
        pred_vec = sim_scores>=threshold
        accuracies[i] = np.mean(pred_vec==labels)

    argmax = np.argmax(accuracies)
    accuracy = accuracies[argmax]
    threshold = np.mean(thresholds[accuracies==accuracy])
    return accuracy, threshold


# -

def evaluate(model, dataset, metric='cosine'):
    model.to(device)
    model.eval()

    sim_scores = []
    labels = []

    # Move the model to evaluation mode and set the data types once
    with torch.no_grad():
        for data in dataset:
            img = data[0].unsqueeze(0)
            img2 = data[1].unsqueeze(0)
            labels.append(data[2])

        # Batch processing: Stack the images for efficient computation
        img = torch.cat([sample[0].unsqueeze(0) for sample in dataset]).to(device).type(torch.float32)
        img2 = torch.cat([sample[1].unsqueeze(0) for sample in dataset]).to(device).type(torch.float32)

        # Compute the representations for all samples in the dataset
        x = model(img)
        y = model(img2)

        # Calculate similarity scores for all pairs in the dataset
        sim_scores = similarity_score(x, y, metric)
        if len(sim_scores.shape)>1:
            sim_scores = sim_scores.diagonal()

    return labels, sim_scores


def plot_roc(far, tar, thresholds):
    auc_score=auc(far,tar)
    plt.plot(far, tar, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def tar_at_far(tar,far):
    a=np.interp(.1,far,tar)
    b=np.interp(.01,far,tar)
    c=np.interp(.001,far,tar)
    d=np.interp(.0001,far,tar)
    print(f'''TAR@FAR:
    1%: {a:.4f}
  0.1%: {b:.4f}
 0.01%: {c:.4f}
0.001%: {d:.4f}''')
    return a,b,c,d


def accuracy(labels, sim_scores, thresholds):
    accuracy_scores = []
    for thresh in thresholds:
        accuracy_scores.append(accuracy_score(labels, [m > thresh for m in sim_scores]))

    accuracies = np.array(accuracy_scores)
    max_accuracy = accuracies.max() 
    max_accuracy_threshold =  thresholds[accuracies.argmax()]
    print(f'Accuracy: {max_accuracy}, Threshold: {max_accuracy_threshold}')
    return max_accuracy, max_accuracy_threshold


def visualize(labels, sim_scores):
    far,tar,thresholds=roc_curve(labels,sim_scores)
    plot_roc(far,tar,thresholds)
    tar_at_far(tar,far)


# +
def save_output(output, name):
    try:
        data = load_output()
    except:
        data = dict()
    data[name] = output
    with open('recall.pkl', 'wb') as file:
        pickle.dump(data, file)
        
def load_output():
    with open('recall.pkl', 'rb') as file:
        return pickle.load(file)


# +
def compute_map_at_r(labels,label_predictions):
    """
    labels : [num_samples] (target labels)
    label_predictions : [number of samples x k] (k predicted labels/neighbours)
    """
    assert label_predictions.dim() == 2
    assert labels.size(0) == label_predictions.size(0)
    assert labels.dtype == label_predictions.dtype

    device = labels.device

    ap_at_r = 0
    for actual, predictions in zip(labels, label_predictions):
        truths = (actual == predictions)
        r = truths.sum()

        if r > 0:
            tp_pos = torch.arange(1, r + 1, device=device)[truths[:r] > 0]
            ap_at_r += torch.div((torch.arange(len(tp_pos), device=device) + 1), tp_pos).sum() / r

    return ap_at_r / len(labels)

def mapr(model, model_name, loader):
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(enumerate(loader), total=len(loader))
        progress_bar.set_description(
            'EVALUATING'
        )
        labels = []
        if model_name == 'PA':
            embeddings = []
            for _, (x_batch, y_batch) in progress_bar:
                model_output = model(x_batch.squeeze().to(device))
                embeddings.append(model_output)
                labels.append(y_batch.squeeze().to(device))
    
            embeddings = torch.cat(embeddings)  # (number of samples, embedding size)
            similarity_mat = F.linear(embeddings, embeddings).to(device)
        else:
            embeddings = [[], []]
            for _, (x_batch, y_batch) in progress_bar:
                model_output = model(x_batch.squeeze().to(device))
                embeddings[0].append(model_output[0])  # mu
                embeddings[1].append(model_output[1])  # sigma
                labels.append(y_batch.squeeze().to(device))
    
            embeddings = torch.cat(embeddings[0])
            similarity_mat = F.linear(embeddings, embeddings).to(device)
  
        labels = torch.cat(labels)  # (number of samples)
  
        sorted_similarity_mat = similarity_mat.sort(descending=True)
  
        # (n, m) tensor where the tensor at index n contains the sorted similarity scores of the most similar samples
        # with the sample at index n, excluding the similarity of a sample with itself. n = m = number of samples.
        ranked_similar_samples_scores = sorted_similarity_mat[0][:, 1:]
  
        # (n, m) tensor where the tensor at index n contains the sorted indices of the most similar samples
        # with the sample at index n, excluding the similarity of a sample with itself. n = m = number of samples.
        ranked_similar_samples_indices = sorted_similarity_mat[1][:, 1:]
  
        ranked_similar_labels = labels[ranked_similar_samples_indices]
  
        map_at_r = compute_map_at_r(labels, ranked_similar_labels)
        print(f'MAP@R: {map_at_r*100:.4f}')

        return map_at_r
