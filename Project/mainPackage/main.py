import numpy as np
import os
import open3d as o3d
from matplotlib import pyplot as plt
from natsort import natsorted
from sklearn.cluster import DBSCAN
import natsort
import ImageInfo as im


class Vehicle:
    def __init__(self, cluster, last_position):
        self.cluster = cluster
        self.last_position = last_position
        self.velocity = np.zeros(3)


def main(imgFiles, pointCloudFiles):
    imageInfo = []
    files = os.listdir(pointCloudFiles)
    files = natsorted(files)
    median_cloud = findMedianCloud(files, pointCloudFiles)
    lastCloud = o3d.geometry.PointCloud()
    points = np.zeros((0, 3))
    lastCloud.points = o3d.utility.Vector3dVector(points)
    vehicles = []

    for filename in files:
        print(pointCloudFiles + "\\" + filename)
        lastCloud, vehicles = createVehicleInfo((pointCloudFiles + "\\" + filename), imageInfo, median_cloud, lastCloud, vehicles)

def findMedianCloud(files,pointCloudFiles):
    pointClouds = []
    x = 0
    for filename in files:
        pointCloud = o3d.io.read_point_cloud(pointCloudFiles + "\\" + filename)
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
    return median_cloud


def createVehicleInfo(file, imageInfo, median_cloud, lastCloud, vehicles):
    lastCloud, vehicles = FillVehics(
        file, imageInfo, median_cloud, lastCloud, vehicles
    )
    return lastCloud, vehicles


def FillVehics(img, imageInfo, median_cloud, lastCloud, vehicles):
    clusters, lastCloud = ClusterLidar(img, median_cloud, lastCloud, vehicles)
    # print("clusters: {}".format(clusters))

    updated_vehicles = []
    for cluster in clusters:
        closest_vehicle = findClosestVehicle(cluster, vehicles)
        if closest_vehicle is not None:
            velocity = cluster.mean(axis=0) - closest_vehicle.last_position
            if 0.1 < velocity < 0.5:
                closest_vehicle.velocity = velocity
                updated_vehicles.append(closest_vehicle)
        else:
            new_vehicle = Vehicle(cluster, cluster.mean(axis=0))
            updated_vehicles.append(new_vehicle)

    return lastCloud, updated_vehicles


def findClosestVehicle(cluster, vehicles):
    min_distance = float("inf")
    closest_vehicle = None

    for vehicle in vehicles:
        distance = np.linalg.norm(cluster.mean(axis=0) - vehicle.last_position)
        if distance < min_distance:
            min_distance = distance
            closest_vehicle = vehicle

    return closest_vehicle


# create clusters to count and track the cars
# measure velocity from the movement of one frame to the next
# compare based on the closest cluster to the position of the cluster in the previous frame
def ClusterLidar(file, median_cloud, last_cloud, vehicles):
    pcd = o3d.io.read_point_cloud(file)
    point_cloud = np.asarray(pcd.points)
    points = remove_matching_points(point_cloud, median_cloud)
    if len(last_cloud.points)>0:
        points = remove_matching_points(points,np.asarray(last_cloud.points))
    print(len(points))
    clusters = np.array(0)
    if(len(points>0)):
        clustering = DBSCAN(eps=.5, min_samples=20).fit(points)
        cluster_labels = clustering.labels_
        visualize_clusters(points, cluster_labels, vehicles)
        unique_labels = set(cluster_labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for cluster, color in zip(unique_labels, colors):
            clusters = points[cluster_labels == cluster]
    return clusters, pcd


def remove_matching_points(pointcloud, mediancloud):
    tolerance = 1e-3
    rounded_pc1 = np.around(pointcloud, decimals=int(-np.log10(tolerance)))
    rounded_pc2 = np.around(mediancloud, decimals=int(-np.log10(tolerance)))

    pc1_set = {tuple(point) for point in rounded_pc1}
    pc2_set = {tuple(point) for point in rounded_pc2}

    # Find common points within the specified tolerance
    matching_points = pc1_set.intersection(pc2_set)
    matching_points = np.array(list(matching_points))

    # Find indices of matching points in each point cloud
    indices_pc1 = np.array([i for i, point in enumerate(rounded_pc1) if tuple(point) in matching_points])
    #indices_pc2 = np.array([i for i, point in enumerate(rounded_pc2) if tuple(point) in matching_points])

    # Remove matching points from both point clouds
    filtered_pc1 = np.delete(pointcloud, indices_pc1, axis=0)

    return filtered_pc1


def visualize_clusters(points, cluster_labels, vehicles):
    # Visualize clusters
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = set(cluster_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for cluster, color in zip(unique_labels, colors):
        cluster_points = points[cluster_labels == cluster]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=[color], marker='o', s=20)


    print("lenvehicles: {}".format(len(vehicles)))
    if len(vehicles) < 30:
        for vehicle in vehicles:
            print("vehicles: {}".format(len(vehicles)))
            cluster = vehicle.cluster
            ax.scatter(cluster[0], cluster[1], cluster[2], c = ["black"], marker='x', s=20)

    plt.show()

if __name__ == "__main__":
    # LOAD Folders
    imgFiles = "dataset\Images"
    pointCloudFiles = "dataset\PointClouds"
    main(imgFiles, pointCloudFiles)
