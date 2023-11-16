import numpy as np
import os
import open3d as o3d
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import ImageInfo.ImageInfo as im


def main(imgFiles, pointCloudfiles):
    imageInfo = []
    pointClouds = []
    x = 0
    for filename in os.listdir(pointCloudfiles):
        pointCloud = o3d.io.read_point_cloud(pointCloudfiles + "\\" + filename)
        points = np.asarray(pointCloud.points)
        pointClouds.append(points)
        if (x >= 70):
            break
        x += 1
    all_points = np.vstack(pointClouds)
    max_points = max(len(pc) for pc in pointClouds)
    median_cloud = np.zeros((max_points, 3))  # Assuming 3 dimensions (x, y, z)

    for i in range(max_points):
        points_at_index = np.array([pc[i] for pc in pointClouds if len(pc) > i])
        median_point = np.median(points_at_index, axis=0) if len(points_at_index) > 0 else np.zeros(3)
        median_cloud[i] = median_point

    median_cloud = median_cloud[~np.all(median_cloud == 0, axis=1)]
    fig = plt.figure(figsize=(24, 24))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(median_cloud[:,0], median_cloud[:,1], median_cloud[:,2], marker='o', s=20)
    plt.show()
    for filename in os.listdir(pointCloudfiles):
        print(pointCloudfiles + "\\" + filename)
        vehicleInfo = createVehicleInfo((pointCloudfiles + "\\" + filename), imageInfo)


def FillVehics(img, imageInfo):
    clusters = ClusterLidar(img)
    for clusts in clusters:
        temp = im.ImageInfo(clusts)
        # temp.SetPosition(clusts.)
        imageInfo.append(temp)


# create clusters to count and track the cars
# measure velocity from the movement of one frame to the next
# compare based on the closest cluster to the position of the cluster in the previous frame
def ClusterLidar(file):
    pcd = o3d.io.read_point_cloud(file)
    points = np.asarray(pcd.points)
    print(len(points))

    clusters = np.array(0)
    clustering = DBSCAN(eps=.5, min_samples=20).fit(points)
    cluster_labels = clustering.labels_
    visualize_clusters(points, cluster_labels)
    unique_labels = set(cluster_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for cluster, color in zip(unique_labels, colors):
        clusters = points[cluster_labels == cluster]

    return clusters


def createVehicleInfo(file, imageInfo):
    fillVehics = FillVehics(file, imageInfo)

    # call functions to set each element of the vehic class


def visualize_clusters(points, cluster_labels):
    # Visualize clusters
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = set(cluster_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for cluster, color in zip(unique_labels, colors):
        cluster_points = points[cluster_labels == cluster]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=[color], marker='o', s=20)
        plt.show()


if __name__ == "__main__":
    # LOAD Folders
    imgFiles = "F:\\computer vision final project\\traffic-light\\dataset\\Images"
    print(imgFiles)
    pointCloudFiles = "F:\\computer vision final project\\traffic-light\\dataset\PointClouds"
    main(imgFiles, pointCloudFiles)
