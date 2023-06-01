import os
import json
import ujson

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

record_path = "/mnt/disk1/naver/hieunguyen/provably_fl/fedtask/cifar10_cnum100_dist8_skew0.1_seed0/record/fedprob_N_0.25Mresnet18_R300_B10_E5_LR0.0500_P0.10_S0_LD-0.002_WD0.000_DR0.00_AC99999.00.json"
record_task = os.path.join("./figure", record_path.split(sep="/")[-3])
figure_path = os.path.join(record_task, record_path.split(
    sep="/")[-1].replace(".json", ""))

data_path = record_path.split(sep="record")[0] + "data.json"

if not os.path.exists(record_task):
    os.makedirs(record_task)

if not os.path.exists(figure_path):
    os.makedirs(figure_path)

record = json.load(open(record_path, "r"))

# Process record file
server_cert_acc = np.array(record["server_certify_acc"])
clients_cert_acc = record["client_certify_acc"]

clients = list(clients_cert_acc.keys())

cert_acc = []
average_client_cert_acc = 0

for idx in clients:
    client_cert_acc = np.array(clients_cert_acc[idx])
    cert_acc.append(client_cert_acc)

client_cert_acc = np.array(cert_acc)
server_cert_acc = np.expand_dims(server_cert_acc, axis=0)
# mean_client_cert_acc = np.mean(client_cert_acc, axis=0)


client_data = ujson.load(open(data_path, "r"))
client_vols = []
for name in client_data["client_names"]:
    volume = len(client_data[name]['dtrain']['x'])
    client_vols.append(volume)

# super_small_client = np.where(client_vols <= 30)
client_vols = np.array(client_vols) / sum(client_vols)

mean_weight_client_acc = np.matmul(
    client_cert_acc.T, np.expand_dims(client_vols, axis=0).T)

# Analyze

# Cosine similarity compare between server and client
similarity = cosine_similarity(server_cert_acc, client_cert_acc)
plt.plot(range(100), similarity.T)
# plt.title("Cosine similarity between server and clients")
plt.xlabel("client id")
plt.ylabel("Similarity")
plt.savefig(os.path.join(figure_path, "heatmap.jpg"))


# Upper and lower bar
plt.clf()
lolims = - np.expand_dims(np.min(client_cert_acc, axis=0),
                          axis=0) + mean_weight_client_acc.T
uplims = np.expand_dims(np.max(client_cert_acc, axis=0),
                        axis=0) - mean_weight_client_acc.T
bound = np.concatenate((lolims, uplims), axis=0)

plt.errorbar(x=np.arange(0, 1.6, 0.1),
             y=mean_weight_client_acc.flatten(), yerr=bound, label="Client")
plt.plot(np.arange(0, 1.6, 0.1), server_cert_acc.flatten(), label="Server")
plt.xlim(right=1.5)
# plt.ylim(bottom=0)
# plt.title("Certify acc vs radius")
plt.xlabel("radius")
plt.ylabel("Certified accuracy")
plt.legend()
plt.savefig(os.path.join(figure_path, "erro_bar.jpg"))


# Training loss and accuracy
plt.clf()
train_loss = record["train_losses"]
# test_accs = record["test_accs"]
plt.plot(train_loss, label="Train loss")
# plt.plot(test_accs, label="Test acc")
plt.legend()
plt.savefig(os.path.join(figure_path, "loss_acc.jpg"))


# Plot distribution of data
plt.clf()
dtest = np.array(client_data["dtest"]["y"])
hist, bins = np.histogram(dtest, 10, [0, 10])
hist = hist / hist.sum()
client_hist = hist
print(hist)

left = 0
for i in range(10):
    plt.barh(0, hist[i], left=left, label=i)
    left = left + hist[i]

dtrain = np.zeros((10, ))
for name in client_data["client_names"]:
    dclient = np.array(client_data[name]['dvalid']['y'])
    dclient, bins = np.histogram(dclient, 10, [0, 10])
    dtrain = dtrain + dclient

hist = dtrain / dtrain.sum()
server_hist = hist
print(hist)

print(np.linalg.norm(client_hist - server_hist))

left = 0
for i in range(10):
    plt.barh(1, hist[i], left=left, label=i)
    left = left + hist[i]

# plt.legend()
plt.savefig(os.path.join(figure_path, "data_distribution.jpg"))


# New aggregate
client_dist = []
for name in client_data['client_names']:
    y_label = np.array(client_data[name]['dvalid']['y'])
    hist, bins = np.histogram(y_label, 10, [0, 10])
    hist = hist / np.sum(hist)
    client_dist.append(hist)
client_dist = np.array(client_dist)
server_dist, bins = np.histogram(dtest, 10, [0, 10])
server_dist = server_dist / np.sum(server_dist)
similarity = cosine_similarity(client_dist, server_dist.reshape(1, -1))
similarity = similarity / np.sum(similarity)
new_agg_cert = np.matmul(client_cert_acc.T, similarity)

plt.clf()
plt.plot(np.arange(0, 1.6, 0.1), mean_weight_client_acc, label="Weight Average")
plt.plot(np.arange(0, 1.6, 0.1), server_cert_acc.T, label="Server")
plt.plot(np.arange(0, 1.6, 0.1), new_agg_cert, label="New aggregate")
plt.xlabel("radius")
plt.ylabel("certified accuracy")
plt.legend()
plt.savefig(os.path.join(figure_path, "new_agg.jpg"))


# Plot most different Client
similarity = cosine_similarity(server_cert_acc, client_cert_acc)
sorted_indices = similarity.argsort()
most_diff_client = sorted_indices[0][0]

plt.clf()
plt.plot(np.arange(0, 1.6, 0.1), server_cert_acc.T, label="Server")
plt.plot(np.arange(0, 1.6, 0.1),
         client_cert_acc[most_diff_client], label="Most Different Client")
plt.legend()
plt.savefig(os.path.join(figure_path, "most_diff.jpg"))


# New aggregate
client_dist = []
client_entropy = []
for name in client_data['client_names']:
    y_label = np.array(client_data[name]['dvalid']['y'])
    hist, bins = np.histogram(y_label, 10, [0, 10])
    hist = hist / np.sum(hist)
    hist = hist + 1e-8
    client_dist.append(hist)
    entropy = -np.sum(hist / np.sum(hist) * np.log2(hist / np.sum(hist)))
    client_entropy.append(entropy)


client_entropy = [entropy / sum(client_entropy) for entropy in client_entropy]
client_entropy = np.array(client_entropy)
# print(f"Entropy: {client_entropy}")
# client_entropy = client_entropy * \
#     np.array(client_vols).reshape(client_entropy.shape)
client_entropy = client_entropy / np.sum(client_entropy)


server_dist, bins = np.histogram(dtest, 10, [0, 10])
server_dist = server_dist / np.sum(server_dist)
tien_cert = np.matmul(client_cert_acc.T, client_entropy)

plt.clf()
plt.plot(np.arange(0, 1.6, 0.1), server_cert_acc.T, label="Server")
plt.plot(np.arange(0, 1.6, 0.1), mean_weight_client_acc, label="Weight Average")
plt.plot(np.arange(0, 1.6, 0.1), new_agg_cert, label="Cosine aggregate")
plt.plot(np.arange(0, 1.6, 0.1), tien_cert, label="Entropy aggregate")
plt.xlabel("radius")
plt.ylabel("certified accuracy")
plt.legend()
plt.savefig(os.path.join(figure_path, "new_agg_v2.jpg"))


# Upper bound
client_dist = []
for name in client_data['client_names']:
    y_label = np.array(client_data[name]['dvalid']['y'])
    hist, bins = np.histogram(y_label, 10, [0, 10])
    hist = hist / np.sum(hist)
    client_dist.append(hist)
client_dist = np.array(client_dist)
server_dist, bins = np.histogram(dtest, 10, [0, 10])


server_dist = server_dist / np.sum(server_dist)
all_client_dist = np.matmul(client_vols.T, client_dist)

epsilon = 0.01

client_cert_accs = [record['client_certify_acc'][client]
                    for client in record['client_certify_acc'].keys()]
client_cert_accs = np.array(client_cert_accs)
all_client_accs = np.matmul(client_cert_accs.T, client_vols)
client_class_accs = np.matmul(np.expand_dims(
    all_client_accs, axis=0).T, np.expand_dims(all_client_dist, axis=0))

std = np.std(client_class_accs, axis=1) + 1e-8
class_means = np.expand_dims(np.mean(client_class_accs, axis=1), axis=0).T

diff = np.sum(((client_class_accs - class_means) *
               client_class_accs), axis=1) / (std)
upper = all_client_accs + diff
lower = all_client_accs - diff
server_cert_acc = np.array(record['server_certify_acc'])

plt.clf()
plt.plot(np.arange(0, 1.6, 0.1), server_cert_acc, label="Server")
plt.plot(np.arange(0, 1.6, 0.1), upper, label="Upper bound")
plt.plot(np.arange(0, 1.6, 0.1), lower, label="Lower bound")
plt.legend()
plt.xlabel("test radius")
plt.ylabel("certified accuracy")
plt.savefig(os.path.join(figure_path, 'upper.jpg'))


# Bound v2
per_client_accuracy = client_cert_accs.T
epsilon = 0.01
mean_accuracy = per_client_accuracy.mean(axis=1, keepdims=True)
std = np.sqrt(np.sum((per_client_accuracy - mean_accuracy)
              ** 2, axis=1, keepdims=True)) + 1e-8
diff = np.sum(((per_client_accuracy - mean_accuracy) * per_client_accuracy)
              * per_client_accuracy, axis=1, keepdims=True) / std * epsilon

upper = mean_weight_client_acc.squeeze() + diff.squeeze()
lower = mean_weight_client_acc.squeeze() - diff.squeeze()

print(mean_accuracy.shape)
print(std.shape)
print(diff.shape)
print(mean_weight_client_acc.shape)

plt.clf()
plt.plot(np.arange(0, 1.6, 0.1), server_cert_acc, label="Server")
plt.plot(np.arange(0, 1.6, 0.1), upper, label="Upper")
plt.plot(np.arange(0, 1.6, 0.1), lower, label="Lower")
# plt.plot(assume_accuracy[:7], label="Assume accuracy")
# plt.plot(server_cert_acc, label="Client")
plt.legend()
plt.xlabel("test radius")
plt.ylabel("certified accuracy")
plt.savefig(os.path.join(figure_path, "bound_v2.jpg"))
