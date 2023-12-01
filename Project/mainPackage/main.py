import numpy as np
import os
import open3d as o3d
from matplotlib import pyplot as plt
from natsort import natsorted
from sklearn.cluster import DBSCAN


cars = {}

def main(pointCloudFiles):
    files = os.listdir(pointCloudFiles)
    files = natsorted(files)
    median_cloud = findMedianCloud(files, pointCloudFiles)
    filename = 'ground_truth.csv'
    with open(filename, 'w') as file:
        file.write("Frame,Vehicle_ID,Pos_X,Pos_Y,Pos_Z,MVec_X,MVec_Y,MVec_Z,BBox_X_Min,BBox_X_Max,BBox_Y_Min,BBox_Y_Max,BBox_Z_Min,BBox_Z_Max\n")
        for filename in files:
            print(pointCloudFiles + "\\" + filename)
            ClusterLidar((pointCloudFiles + "\\" + filename), median_cloud)
            for car_id, car_info in cars.items():
                Frame = filename[:-4]
                Vehicle_ID = car_id
                Pos_X = car_info['center'][0]
                Pos_Y = car_info['center'][1]
                Pos_Z = car_info['center'][2]
                MVec_X = car_info['MvecX']
                MVec_Y = car_info['MvecY']
                MVec_Z = car_info['MvecZ']
                BBox_X_Min = car_info['coord_min'][0]
                BBox_X_Max = car_info['coord_max'][0]
                BBox_Y_Min = car_info['coord_min'][1]
                BBox_Y_Max = car_info['coord_max'][1]
                BBox_Z_Min = car_info['coord_min'][2]
                BBox_Z_Max = car_info['coord_max'][2]
                file.write(f"{Frame},{Vehicle_ID},{Pos_X},{Pos_Y},{Pos_Z},{MVec_X},{MVec_Y},{MVec_Z},{BBox_X_Min},{BBox_X_Max},{BBox_Y_Min},{BBox_Y_Max},{BBox_Z_Min},{BBox_Z_Max}\n")

#@Param files the list of files.
#@Param point cloud files string to path correctly.
#@return a median cloud of the first 70 frames outlier removal.
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

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    return inlier_cloud

#@Param file the current file being examined
#@Param median_cloud median cloud of the first 70 frames outlier removal.
#@Return list of clusters for velocity and positioning determination.
def ClusterLidar(file, median_cloud):
    pcd = o3d.io.read_point_cloud(file)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size = 0.04)

    print("Statistical oulier removal")
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors = 25, std_ratio = 1)
    inlier_cloud = display_inlier_outlier(voxel_down_pcd, ind)

    point_cloud = np.asarray(inlier_cloud.points)
    points = remove_matching_points(point_cloud, median_cloud)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    clusters = np.array(0)
    if(len(points>0)):
        dbscan = DBSCAN(eps=0.8, min_samples=10)  # Adjust parameters as needed 1,30
        labels = dbscan.fit_predict(point_cloud.points)

        # Get unique cluster labels
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels[unique_labels != -1])  # Exclude noise points (-1 label)
        cluster_positions = []
        if (num_clusters > 0):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for cluster_label in range(num_clusters):
                cluster_indices = np.where(labels == cluster_label)[0]
                if len(cluster_indices) > 0:
                    cluster_points = points[cluster_indices]
                    centroid = np.mean(cluster_points, axis=0)
                    cluster_positions.append(centroid)
                    min_xyz = np.min(cluster_points, axis=0)
                    max_xyz = np.max(cluster_points, axis=0)

                    if max_xyz[0] - min_xyz[0] <= 4 and max_xyz[1] - min_xyz[1] <= 4 and max_xyz[2] <= 5:  # Set this to 5 to remove traffic light noise
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
                        # print(f"CAR {cluster_label}: Center:({(max_xyz[0] - min_xyz[0]):.8f}, {(max_xyz[1] - min_xyz[1]):.8f}, {(max_xyz[2] - min_xyz[2]):.8f})")

                        new_car_center = centroid
                        if cars:
                            similarity_threshold = 1e-2 ## Change this threshold accordingly

                            for car_id, car_info in cars.items():
                                existing_center = car_info['center']

                                if (
                                    abs(existing_center[0] - new_car_center[0]) < similarity_threshold
                                    and abs(existing_center[1] - new_car_center[1]) < similarity_threshold
                                    and abs(existing_center[2] - new_car_center[2]) < similarity_threshold
                                ):
                                    cars[car_id]['MvecX'] = round((existing_center[0] - new_car_center[0])/(1/30),8)
                                    cars[car_id]['MvecY'] = round((existing_center[1] - new_car_center[1])/(1/30),8)
                                    cars[car_id]['MvecZ'] = round((existing_center[2] - new_car_center[2])/(1/30),8)
                                    cars[car_id]['center'] = tuple(round(coord, 8) for coord in new_car_center)
                                    cars[car_id]['coord_min'] = min_xyz
                                    cars[car_id]['coord_max'] = max_xyz
                                    break
                            else:
                                car_id = len(cars) + 1
                                cars[car_id] = {'center': tuple(round(coord, 8) for coord in new_car_center),
                                                'MvecX': 0.0,
                                                'MvecY': 0.0,
                                                'MvecZ': 0.0,
                                                'coord_min': min_xyz,
                                                'coord_max': max_xyz}
                        else:
                            cars[1] = {'center': tuple(round(coord, 8) for coord in new_car_center),
                                       'MvecX': 0.0,
                                       'MvecY': 0.0,
                                       'MvecZ': 0.0,
                                       'coord_min': min_xyz,
                                       'coord_max': max_xyz}

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
    return clusters

#@Param pointcloud point cloud you want to remove points from
#@Param mediancloud point cloud you want to remove from the other.
#@Return The difference of the point clouds
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

#@Param points nparray of points
#@Param cluster_lables labels for the clusters
#@Return vehicles the clusters that represent vehicles.
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
    imgFiles = "F:\\Computer Vision\\traffic-light\\ryan\\dataset\\Images"
    pointCloudFiles = "F:\\Computer Vision\\traffic-light\\ryan\\dataset\\PointClouds"
    main(pointCloudFiles)
