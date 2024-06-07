import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import robotpy_apriltag as ra



def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

#direct linear transform
def DLT(P1, P2, point1, point2):

    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))
    # print('A: ')
    # print(A)

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)

    #print('Triangulated point: ')
    #print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]

def read_camera_parameters(camera_id):

    inf = open('camera_parameters/c' + str(camera_id) + '.dat', 'r')

    cmtx = []
    dist = []

    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    return np.array(cmtx), np.array(dist)

def read_rotation_translation(camera_id, savefolder = 'camera_parameters/'):

    inf = open(savefolder + 'rot_trans_c'+ str(camera_id) + '.dat', 'r')

    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)

    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)

    inf.close()
    return np.array(rot), np.array(trans)

def _convert_to_homogeneous(pts):
    pts = np.array(pts)
    if len(pts.shape) > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis = 1)
    else:
        return np.concatenate([pts, [1]], axis = 0)

def get_projection_matrix(camera_id):

    #read camera parameters
    cmtx, dist = read_camera_parameters(camera_id)
    rvec, tvec = read_rotation_translation(camera_id)

    #calculate projection matrix
    P = cmtx @ _make_homogeneous_rep_matrix(rvec, tvec)[:3,:]
    return P

def write_keypoints_to_disk(filename, kpts):
    fout = open(filename, 'w')

    for frame_kpts in kpts:
        for kpt in frame_kpts:
            if len(kpt) == 2:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ')
            else:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ' + str(kpt[2]) + ' ')

        fout.write('\n')
    fout.close()

# Function to read in an object file that was generated from an object consisting only of triangles. The object file to read in should only
# contain information about the vertices and the faces of the object.   --> changed by PG
def read_obj(file):
    vertices = []
    faces = []
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertex = list(map(float, line.strip().split()[1:]))
                vertices.append(vertex)
            elif line.startswith('f '):
                face = [int(x.split('/')[0]) - 1 for x in line.strip().split()[1:]]
                faces.append(face)
    return np.array(vertices), np.array(faces)

# Function to plot an object file at a certain position in a given 3D plot.   --> changed by PG
def plot_obj(camera_file='camera03.obj', rot_trans='rot_trans_c0.dat', ax_=plt.figure(), cam_name='cam'):
    vertices, faces = read_obj(camera_file)

    rot, trans = read_rotation_translation(rot_trans)

    coord_rot = np.array([[0, 0, 1],
                         [0, 1, 0],
                         [-1, 0, 0]])
    size = np.array([[10, 0, 0],
                    [0, 10, 0],
                    [0, 0, 10]])
    
    # vertices = np.dot(vertices, size)
    vertices = np.dot(vertices, rot) - trans.T
    vertices = np.dot(vertices, coord_rot)

    cam_name = 'Camera ' + cam_name
    
    ax_.plot_trisurf(vertices[:,0], vertices[:,1], faces, vertices[:,2],  edgecolor='none')
    ax_.text(vertices[0,0]+10.0, vertices[0,1], vertices[0,2], str(cam_name) )

# Function to return the pose estimation of a april tag in a certain frame number of a certain video.   --> changed by PG
def estimate_pose_of_aprilTag(frame_number, video_name):

    # reads out the intrinsic camera parameters of camera 0
    cmtx, dist = read_camera_parameters('0')
    cx0 = cmtx[0,0]
    cy0 = cmtx[1,1]
    fx0 = cmtx[0,2]
    fy0 = cmtx[1,2]

    video = cv.VideoCapture(video_name)

    # reads out a certain frame of the loeaded video
    video.set(cv.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()
    
    # saves a grayscale image of the read out frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # initialises an april tag detector that detects april tags of the family "36h11"
    detector = ra.AprilTagDetector()
    detector.addFamily('tag36h11')

    # save the detected april tags in the variable "detections"
    detections = detector.detect(gray)

    # configures and initialises an april tag pose estimator
    pose_estimator_config = ra.AprilTagPoseEstimator.Config(tagSize=0.2, cx=cx0, cy=cy0, fx=fx0, fy=fy0)
    pose_estimator = ra.AprilTagPoseEstimator(pose_estimator_config)

    # saves the pose estimations of the before detected april tags in the list "estimations"
    estimations = []
    for i in range(len(detections)):
        estimations.append([pose_estimator.estimate(detections[i]), detections[i].getId()])
    
    return estimations

# Returns the plane coefficients of a april tag lying in the plane. Input needs to be a pose estimated april tag
# which can be generated with the function estimate_pose_of_aprilTag().   --> changed by PG
def get_plane_coeff(estimation:ra.AprilTagPoseEstimate):
    # extract the rotation in radians from the pose estimation of the april tag for each coordinate direction
    rotVec = estimation.rotation().getQuaternion().toRotationVector()
    rotX = rotVec[0]
    rotY = rotVec[1]
    rotZ = rotVec[2]

    # define the rotation matrices for each coordinate direction and combine them into "R"
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(rotX), -np.sin(rotX)],
                    [0, np.sin(rotX), np.cos(rotX)]])
    
    R_y = np.array([[np.cos(rotY), 0, np.sin(rotY)],
                    [0, 1, 0],
                    [-np.sin(rotY), 0, np.cos(rotY)]])
    
    R_z = np.array([[np.cos(rotZ), -np.sin(rotZ), 0],
                    [np.sin(rotZ), np.cos(rotZ), 0],
                    [0, 0, 1]])
    
    R = np.dot(R_z, np.dot(R_y, R_x))

    # define a unit vector
    unitVec = np.array([1, 0, 0])

    # Rotate the unit vector by the rotation matrix "R" to get the coefficients of the plane in which the april tag lies.
    # (the coeffs vector is also the normal vector of the plane)
    coeffs = np.dot(R, unitVec)

    return coeffs

# Returns the distance between a given point and a given plane.   --> changed by PG
def distance_to_plane(point, plane_coeffs):
    # extract plane coefficients
    a, b, c = plane_coeffs

    # extract point coordinates
    x, y, z = point

    # calculates the distance between the plane and the point
    distance = abs(a * x + b * y + c * z - (a**2 + b**2 + c**2) ** 0.5) / (a**2 + b**2 + c**2) ** 0.5

    return distance

if __name__ == '__main__':

    P2 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)
