import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class Clustering():
    def __init__(self, n_clusters=2, alpha=0.7, beta=0.25):
        super(Clustering, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.beta = beta

    def clusters(self, vid_seg, labels):
        scaler = StandardScaler()
        normalize_data = scaler.fit_transform(vid_seg.detach().cpu().numpy())

        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(normalize_data)

        cluster_centers = kmeans.cluster_centers_
        cluster_centers_tensor = torch.tensor(cluster_centers)
        normalized_data = torch.tensor(normalize_data)

        if labels == 0:
            cluster_loss = self.normal_loss(cluster_centers_tensor, normalized_data)
        else:
            cosine_sim_cc = F.cosine_similarity(cluster_centers_tensor[0].unsqueeze(0), 
                                               cluster_centers_tensor[1].unsqueeze(0))
            selected_features_cluster1 = []
            selected_features_cluster2 = []
            c1 = 0
            c2 = 0
            
            for i, point in enumerate(normalized_data):
                cluster_label = kmeans.labels_[i]
                point_tensor = point

                sim_to_cluster_center = 1 - F.cosine_similarity(point_tensor, 
                                                              cluster_centers_tensor[cluster_label], 
                                                              dim=0)

                if cluster_label == 0 and c1 == 0:
                    temp = point_tensor
                    c1 += 1
                if cluster_label == 1 and c2 == 0:
                    temp1 = point_tensor
                    c2 += 1

                if cluster_label == 0 and sim_to_cluster_center < self.beta * (1 - cosine_sim_cc):
                    selected_features_cluster1.append(point_tensor)
                elif cluster_label == 1 and sim_to_cluster_center < self.beta * (1 - cosine_sim_cc):
                    selected_features_cluster2.append(point_tensor)

            if len(selected_features_cluster1) == 0:
                selected_features_cluster1 = [temp]
            if len(selected_features_cluster2) == 0:
                selected_features_cluster2 = [temp1]

            cluster_loss = self.anomaly_loss(cluster_centers_tensor, 
                                           selected_features_cluster1, 
                                           selected_features_cluster2)

        return cluster_loss

    def normal_loss(self, cluster_centers, normalized_data):
        all_lcn = []
        device = cluster_centers.device
        mean_center = torch.mean(cluster_centers, dim=0).to(device)

        for feature in normalized_data:
            temp = 1 - F.cosine_similarity(feature.reshape(1,-1), mean_center.reshape(1,-1))
            all_lcn.append(temp)
        l_c_n = (torch.mean(torch.stack(all_lcn).squeeze(1)))

        return l_c_n

    def anomaly_loss(self, cluster_centers, selected_features_cluster1, selected_features_cluster2):
        lcc1 = []
        lcc2 = []
        lcd1 = []
        lcd2 = []

        for features in selected_features_cluster1:
            similarity = 1 - torch.cosine_similarity(features.unsqueeze(0), 
                                                     cluster_centers[0].unsqueeze(0))
            lcc1.append(similarity)
            similarity2 = 1 + torch.cosine_similarity(features.unsqueeze(0), 
                                                     cluster_centers[1].unsqueeze(0))
            lcd1.append(similarity2)

        for feature in selected_features_cluster2:
            similarity3 = 1 - torch.cosine_similarity(feature.unsqueeze(0), 
                                                     cluster_centers[1].unsqueeze(0))
            lcc2.append(similarity3)
            similarity4 = 1 + torch.cosine_similarity(feature.unsqueeze(0), 
                                                     cluster_centers[0].unsqueeze(0))
            lcd2.append(similarity4)

        lcc = torch.mean(torch.stack(lcc1 + lcc2))
        lcd = torch.mean(torch.stack(lcd1 + lcd2))
        l_c_a = self.alpha * lcc + (1 - self.alpha) * lcd
        return l_c_a