import numpy as np
import os
import open3d as o3d
from matplotlib import pyplot as plt
from natsort import natsorted
from sklearn.cluster import DBSCAN
import natsort
import ImageInfo as im


def main(imgFiles, pointCloudFiles):
    imageInfo = []
    files = os.listdir(pointCloudFiles)
    files = natsorted(files)
    median_cloud = findMedianCloud(files, pointCloudFiles)
    lastCloud = o3d.geometry.PointCloud()
    points = np.zeros((0, 3))
    lastCloud.points = o3d.utility.Vector3dVector(points)
    for filename in files:
        print(pointCloudFiles + "\\" + filename)
        lastCloud = createVehicleInfo((pointCloudFiles + "\\" + filename), imageInfo, median_cloud, lastCloud)

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



def FillVehics(img, imageInfo, median_cloud,lastCloud):
    clusters,lastCloud = ClusterLidar(img,median_cloud,lastCloud)
    return lastCloud
    #for clusts in clusters:
        # temp = im.ImageInfo(clusts, median_cloud)
        # temp.SetPosition(clusts.)
        #imageInfo.append(temp)

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    # print("{} outliers. That is {}%".format(len(outlier_cloud.points), (len(outlier_cloud.points)/(len(outlier_cloud.points)+len(inlier_cloud.points)))))
    
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    return inlier_cloud

# create clusters to count and track the cars
# measure velocity from the movement of one frame to the next
# compare based on the closest cluster to the position of the cluster in the previous frame
def ClusterLidar(file, median_cloud, last_cloud):
    pcd = o3d.io.read_point_cloud(file)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size = 0.04)

    print("Statistical oulier removal")
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors = 25, std_ratio = 1)
    inlier_cloud = display_inlier_outlier(voxel_down_pcd, ind)

    point_cloud = np.asarray(inlier_cloud.points)
    points = remove_matching_points(point_cloud, median_cloud)
    #if len(last_cloud.points)>0:
    #   points = remove_matching_points(points,np.asarray(last_cloud.points))
    #print(len(points))
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    clusters = np.array(0)
    if(len(points>0)):
        #clustering = DBSCAN(eps=.5, min_samples=20).fit(points)
        #cluster_labels = clustering.labels_
        #visualize_clusters(points, cluster_labels)
        #unique_labels = set(cluster_labels)
        #colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        dbscan = DBSCAN(eps=0.8, min_samples=10)  # Adjust parameters as needed 1,30
        labels = dbscan.fit_predict(point_cloud.points)

        # Get unique cluster labels
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels[unique_labels != -1])  # Exclude noise points (-1 label)
        print(num_clusters)

        if (num_clusters > 0):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for cluster_label in range(num_clusters):
                cluster_indices = np.where(labels == cluster_label)[0]
                if len(cluster_indices) > 0:
                    cluster_points = points[cluster_indices]

                    min_xyz = np.min(cluster_points, axis=0)
                    max_xyz = np.max(cluster_points, axis=0)

                    if max_xyz[2] <= 15:  # Set this to 5 to remove traffic light noise
                        edges = [
                            [min_xyz[0], min_xyz[1], min_xyz[2]],
                            [max_xyz[0], min_xyz[1], min_xyz[2]],
                            [max_xyz[0], max_xyz[1], min_xyz[2]],
                            [min_xyz[0], max_xyz[1], min_xyz[2]],
                            [min_xyz[0], min_xyz[1], max_xyz[2]],
                            [max_xyz[0], min_xyz[1], max_xyz[2]],
                            [max_xyz[0], max_xyz[1], max_xyz[2]],
                            [min_xyz[0], max_xyz[1], max_xyz[2]],
                        ]
                        edges = np.array(edges)

                        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], marker='.',
                                label=f'Cluster {cluster_label}')

                        for i, j in zip([0, 1, 2, 3], [1, 2, 3, 0]):
                            ax.plot3D([edges[i, 0], edges[j, 0]], [edges[i, 1], edges[j, 1]], [edges[i, 2], edges[j, 2]],
                                    color='r')

                        for i, j in zip([4, 5, 6, 7], [5, 6, 7, 4]):
                            ax.plot3D([edges[i, 0], edges[j, 0]], [edges[i, 1], edges[j, 1]], [edges[i, 2], edges[j, 2]],
                                    color='r')

                        for i, j in zip([0, 4], [1, 5]):
                            ax.plot3D([edges[i, 0], edges[j, 0]], [edges[i, 1], edges[j, 1]], [edges[i, 2], edges[j, 2]],
                                    color='r')

                        for i, j in zip([2, 6], [3, 7]):
                            ax.plot3D([edges[i, 0], edges[j, 0]], [edges[i, 1], edges[j, 1]], [edges[i, 2], edges[j, 2]],
                                    color='r')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
    plt.show()
    return clusters, pcd


def createVehicleInfo(file, imageInfo, median_cloud,lastCloud):
    lastCloud = FillVehics(file, imageInfo, median_cloud,lastCloud)
    return lastCloud

    # call functions to set each element of the vehic class


def remove_matching_points(pointcloud, mediancloud):
    tolerance = 1e-6
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

if __name__ == "__main__":
    # LOAD Folders
    imgFiles = "dataset\\Images"
    print(imgFiles)
    pointCloudFiles = "dataset\PointClouds"
    main(imgFiles, pointCloudFiles)
