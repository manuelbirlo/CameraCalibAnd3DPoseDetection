import numpy as np
import matplotlib.pyplot as plt
from utils import DLT, plot_obj, estimate_pose_of_aprilTag, get_plane_coeff, distance_to_plane #   --> changed by PG
#plt.style.use('seaborn')   --> changed by PG
import bodypose3d


pose_keypoints = np.array([16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28])

def read_keypoints(filename):
    fin = open(filename, 'r')

    kpts = []
    while(True):
        line = fin.readline()
        if line == '': break

        line = line.split()
        line = [float(s) for s in line]

        line = np.reshape(line, (len(pose_keypoints), -1))
        kpts.append(line)

    kpts = np.array(kpts)
    return kpts


def visualize_3d(p3ds):

    """Now visualize in 3D"""
    torso = [[0, 1] , [1, 7], [7, 6], [6, 0]]
    armr = [[1, 3], [3, 5]]
    arml = [[0, 2], [2, 4]]
    legr = [[6, 8], [8, 10]]
    legl = [[7, 9], [9, 11]]
    body = [torso, arml, armr, legr, legl]
    colors = ['red', 'blue', 'green', 'black', 'orange']

    from mpl_toolkits.mplot3d import Axes3D

    # List for the distances between the left shoulder keypoint and camera 0   --> changed by PG
    distances = []

    # Estimates the pose of the april tags in the 10th frame of the video which was taken to calculate the 3D coordinates of the body keypoints.   --> changed by PG
    at_estimations = estimate_pose_of_aprilTag(10,bodypose3d.input_stream1)
    wall1_est = at_estimations[0][0]
    wall2_est = at_estimations[1][0]

    # Returns coefficients of the wall equation for each wall  --> changed by PG
    wall1_plane_coeff = get_plane_coeff(wall1_est)
    wall2_plane_coeff = get_plane_coeff(wall2_est)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for framenum, kpts3d in enumerate(p3ds):
        if framenum%2 == 0: continue #skip every 2nd frame
        for bodypart, part_color in zip(body, colors):
            for _c in bodypart:
                ax.plot(xs = [kpts3d[_c[0],0], kpts3d[_c[1],0]], ys = [kpts3d[_c[0],1], kpts3d[_c[1],1]], zs = [kpts3d[_c[0],2], kpts3d[_c[1],2]], linewidth = 4, c = part_color)

        # uncomment these if you want scatter plot of keypoints and their indices.
        # for i in range(12):
        #     ax.text(kpts3d[i,0], kpts3d[i,1], kpts3d[i,2], str(i))
        #     ax.scatter(xs = kpts3d[i:i+1,0], ys = kpts3d[i:i+1,1], zs = kpts3d[i:i+1,2])

        # Lines 68 to 76 will calculate the distance of the left shoulder keypoint to camera 0   --> changed by PG
        dist_to_c0 = np.sqrt(kpts3d[1,0]**2 + kpts3d[1,1]**2 + kpts3d[1,2]**2)
        if dist_to_c0 > 3500: # simple filter to avoid too small values
            distances.append(dist_to_c0)
        
        distance_to_wall1 = distance_to_plane([kpts3d[1,0], kpts3d[1,1], kpts3d[1,2]],wall1_plane_coeff)
        distance_to_wall2 = distance_to_plane([kpts3d[1,0], kpts3d[1,1], kpts3d[1,2]],wall2_plane_coeff)
        # print('Distance to Wall 1: '+str(distance_to_wall1))
        # print('Distance to Wall 2: '+str(distance_to_wall2))
        # ax.text(kpts3d[1,0], kpts3d[1,1], kpts3d[1,2], str(kpts3d[1,2]))

        # The following two lines will add the camera object to the visualisation.   --> changed by PG
        plot_obj(rot_trans='0', ax_=ax, cam_name='0')
        plot_obj(rot_trans='1', ax_=ax, cam_name='1')

        # ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_xlim(-2000, 2000)
        ax.set_xlabel('x')
        ax.set_ylim(-2000, 2000)
        ax.set_ylabel('y')
        ax.set_zlim(-2000, 2000)
        ax.set_zlabel('z')
        plt.pause(0.1)
        ax.cla()
    

    # Prints the mean distance of the measured distances between the left shoulder keypoint and the wall 1   --> changed by PG
    distances_mean = np.mean(distances)
    print('distance to camera 0:')
    print(distances_mean)


if __name__ == '__main__':

    p3ds = read_keypoints('kpts_3d.dat')
    visualize_3d(p3ds)
