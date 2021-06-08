# from epipoler_lines import drawlines
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Function to compute the Interest Points
Input : Gray Image
Output : Interest Points, Descriptors
"""
def compute_features(img):
    # detector = cv2.SIFT_create()
    # kp, des = detector.detectAndCompute(img,None)
    detector = cv2.GFTTDetector_create(2000, useHarrisDetector=True)
    descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp = detector.detect(img)
    kp, des = descriptor.compute(img, kp)
    return kp, des


"""
Function to compute the matching points b/w two images
Input : Descriptors of Image1, Descriptors of Image2
Output : List of matched point pairs (type : cv::DMatch)
"""
def compute_matchers(des1, des2):
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2,k=2)
    # filtered_matches = []
    # matched_kp1 = []
    # matched_kp2 = []
    # for i,(m,n) in enumerate(matches):
    #     if m.distance < 0.8*n.distance:
    #         matched_kp2.append(kp2[m.trainIdx].pt)
    #         matched_kp1.append(kp1[m.queryIdx].pt)
    #         filtered_matches.append(cv2.DMatch(m.queryIdx, m.trainIdx, m.distance))
    # matched_kp1 = np.array(matched_kp1)
    # matched_kp2 = np.array(matched_kp2)
    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    mindist = matches[0].distance
    filtered_matches = []
    for match in matches:
        if (match.distance <= max(2*mindist, 50.0)):
            filtered_matches.append(match)
    matched_kp1 = []
    matched_kp2 = []
    for match in matches:
        matched_kp1.append(kp1[match.queryIdx].pt)
        matched_kp2.append(kp2[match.trainIdx].pt)
    matched_kp1 = np.array(matched_kp1)
    matched_kp2 = np.array(matched_kp2)
    return filtered_matches, matched_kp1, matched_kp2


"""
Function to normalize the keypoints such that all the points are within 
the same scale. This is to just stabilize the results
Input : Array of points (Nx2)
Output : Array of normalized points, normalization matrix
"""
def normalize_pts(points):
	centroid = np.mean(points, axis=0)
	rms = np.sqrt(np.sum((points - centroid) ** 2) / points.shape[0])
	norm_factor = np.sqrt(2) / rms
	matrix = np.array([[norm_factor, 0.0, -norm_factor * centroid[0]],
					   [0.0, norm_factor, -norm_factor * centroid[1]],
					   [0.0, 0.0, 1]])
	pointsh = np.row_stack([points.T, np.ones((points.shape[0]),)])
	new_pointsh = (matrix @ pointsh).T
	new_points = new_pointsh[:, :2]
	new_points[:, 0] /= new_pointsh[:, 2]
	new_points[:, 1] /= new_pointsh[:, 2]
	return matrix, new_points


"""
The 8-point algorithm
Input : matched points for image1, matched points for image2 (Nx2)
Output : The fundamental matrix (3x3 matrix F)
"""
def eight_pt_algo(kp1, kp2):
    # normalizing the points
    mat1, kp1 = normalize_pts(kp1)
    mat2, kp2 = normalize_pts(kp2)
    # Creating a matrix to solve x'.T @ F @ x = 0, by rewriting it in
    # the form Ax = 0 and solve for x by performing svd on A.
    # Each row of the A matrix is the Kronecker product of the 
    # matched points in homogeneous form
    A = []
    for i in range(len(kp1)):
        A.append([(kp2[i,0]*kp1[i,0]), (kp2[i,0]*kp1[i,1]), kp2[i,0], 
                  (kp2[i,1]*kp1[i,0]), (kp2[i,1]*kp1[i,1]), kp2[i,1],
                  kp1[i,0], kp1[i,1], 1.0])
    A = np.array(A)
    # perform svd and consider the row of V_transpose or column of V
    # corresponding to the smallest singular value and reshape to 3x3
    # Here, svd results in V_transpose
    _, _, Vt = np.linalg.svd(A)
    f = Vt[-1, :].reshape(3,3)
    # Because F is rank two matrix, forcing the last singular value
    # to 0, so that the matrix is in the space of fundamental matrices
    U, S, Vt = np.linalg.svd(f)
    S[2] = 0
    F = U @ np.diag(S) @ Vt
    # unnormalizing the normalized_F matrix obtained using the normalized
    # points using the normalization matrices
    F = mat2.T @ F @ mat1
    F = F/F[2,2]
    return F


"""
Function to evaluate the estimated F matrix during the RANSAC process using
the Sampson distance. It also calculates the number of inlier points that
satisfy the estimated F.
Inputs : array of matched keypoints for image1 and image2 (Nx2), F matrix for 
which to evaluate, error threshold for the point to be considered inlier
Output : inlier points in image1 and image2 (Nx2), inlier count
"""
def evaluate_F(kp1, kp2, estimate, thresh):
    inlier_count = 0
    img1_inliers = []
    img2_inliers = []
    # converting points of homogeneous coordinates
    kp1_homogeneous = np.column_stack([kp1, np.ones(kp1.shape[0])])
    kp2_homogeneous = np.column_stack([kp2, np.ones(kp2.shape[0])])
    # Sampson's Distance : https://cseweb.ucsd.edu/classes/sp04/cse252b/notes/lec11/lec11.pdf
    # for every point check if it is inlier or not
    for i in range(len(kp1)):
        numerator = (kp2_homogeneous[i] @ estimate @ kp1_homogeneous[i].T)**2
        # epipolar lines
        l2 = estimate @ kp1_homogeneous[i].T
        l1 = estimate.T @ kp2_homogeneous[i].T
        denominator = l1[0] ** 2 + l1[1] ** 2 + l2[0] ** 2 + l2[1] ** 2
        # this error is based on Sampson's distance
        error = numerator/denominator
        # if error less than threshold, consider is as inlier
        if error < thresh:
            img1_inliers.append(kp1[i])
            img2_inliers.append(kp2[i])
            inlier_count += 1
    print("Inlier count : ", inlier_count)
    return (np.array(img1_inliers), np.array(img2_inliers), inlier_count)


"""
The RANSAC algorithm runs for num_iterations and outputs the best fundamental
matrix which is satisfied by most number of points
Input : array of matched keypoints for image1 and image2 (Nx2), max number of
iterations, error_threshold
Output : Best fundamental matrix, inlier points in image1 and image2 (Nx2)
"""
def Ransac(kp1, kp2, num_iterations, thresh):
    best_img1_inlier_pts = None
    best_img2_inlier_pts = None
    best_F_mat = None
    best_inlier_count = 0
    num_pts = len(kp1)
    ite = 0
    while ite < num_iterations:
        # randomly choose 10 points ()
        rand_choice = np.unique(np.random.choice(range(num_pts-1), 8, replace=False))
        rand_kp1 = np.array([kp1[i] for i in rand_choice])
        rand_kp2 = np.array([kp2[i] for i in rand_choice])
        estimated_F = eight_pt_algo(rand_kp1, rand_kp2)
        img1_inlier_pts, img2_inlier_pts, current_inlier_count = evaluate_F(kp1, kp2, estimated_F, thresh)
        if current_inlier_count > best_inlier_count:
            best_inlier_count = current_inlier_count
            best_F_mat = estimated_F
            best_img1_inlier_pts = img1_inlier_pts
            best_img2_inlier_pts = img2_inlier_pts
        ite += 1
    return best_F_mat 



np.random.seed(0)
# K = np.array([[520.0, 0.0, 325.1], 
#               [0.0, 521.0, 249.7],
#               [0.0, 0.0, 1.0]])

# img1 = cv2.imread("1.png", 0)
# kp1, des1 = compute_features(img1)
# img2 = cv2.imread("2.png", 0)
# kp2, des2 = compute_features(img2)
K = np.array([[718.8560,0.0,607.1928],[0.0,718.8560,185.2157],[0.0,0.0,1]])
# img1 = cv2.imread("000000.png", 0)
# kp1, des1 = compute_features(img1)
# img2 = cv2.imread("000001.png", 0)
# kp2, des2 = compute_features(img2)

img1 = cv2.imread("left.jpg", 0)
kp1, des1 = compute_features(img1)
img2 = cv2.imread("right.jpg", 0)
kp2, des2 = compute_features(img2)

matches, matched_kp1, matched_kp2 = compute_matchers(des1, des2)

F = Ransac(matched_kp1, matched_kp2, 50, 10.0)
print(F)
# F, mask = cv2.findFundamentalMat(matched_kp1, matched_kp2, cv2.FM_LMEDS)

def drawlines(img1,lines):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    for r in lines:
        # print(r[0])
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
    return img1


lines1 = cv2.computeCorrespondEpilines(matched_kp2.reshape(-1,1,2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5 = drawlines(img1, lines1)
lines2 = cv2.computeCorrespondEpilines(matched_kp1.reshape(-1,1,2), 1, F)
lines2 = lines2.reshape(-1, 3)
img6 = drawlines(img2, lines2)

E = K.T @ F @ K
U, S, Vt = np.linalg.svd(E)
sigma = 1.0
E = U @ np.diag([sigma, sigma, 0.0]) @ Vt
print(E)
# W = np.array([[0.0,-1.0,0.0], [1.0,0.0,0.0], [0.0,0.0,1.0]])
# U, S, Vt = np.linalg.svd(E)
# R = U @ W.T @ Vt
# print(R)
# t = U @ W @ np.diag(S) @ U.T
# print(t)
_, R, t, mask = cv2.recoverPose(E, matched_kp1, matched_kp2, K)
print(R)
print(t)

E, mask = cv2.findEssentialMat(matched_kp1, matched_kp2, K, cv2.RANSAC)
u, s, vt = np.linalg.svd(E)
# print(u)
# print(s)
print(E)
_, R, t, mask = cv2.recoverPose(E, matched_kp1, matched_kp2, K)
print(R)
print(t)
# W = np.array([[0.0,-1.0,0.0], [1.0,0.0,0.0], [0.0,0.0,1.0]])
# U, S, Vt = np.linalg.svd(E)
# R = U @ W.T @ Vt
# print(R)

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img6)
plt.show()

outmatches = cv2.drawMatches(img1, kp1, img2, kp2, matches, np.array([]))
# print(kp1[matches[0].queryIdx].pt, kp2[matches[0].trainIdx].pt)
cv2.imshow("", outmatches)
cv2.waitKey(0)
cv2.destroyAllWindows()
""" outmatches = cv2.drawMatches(img1, kp1, img2, kp2, matches, np.array([]))
print(kp1[matches[0].queryIdx].pt, kp2[matches[0].trainIdx].pt)
cv2.imshow("", outmatches)
cv2.waitKey(0)
cv2.destroyAllWindows()

out1 = cv2.drawKeypoints(img1, kp1, np.array([]), color=(255,0,0))
out2 = cv2.drawKeypoints(img2, kp2, np.array([]), color=(255,0,0))
# print(np.array(kp1).size(), np.array(des1).size())
# print(np.array(kp1).shape)
cv2.imshow("", out1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow("", out2)
cv2.waitKey(0)
cv2.destroyAllWindows() """



# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# img1 = cv.imread('000000.png',0)  #queryimage # left image
# img2 = cv.imread('000001.png',0) #trainimage # right image
# sift = cv.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# # FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)
# flann = cv.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(des1,des2,k=2)
# pts1 = []
# pts2 = []
# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.8*n.distance:
#         pts2.append(kp2[m.trainIdx].pt)
#         pts1.append(kp1[m.queryIdx].pt)

# pts1 = np.int32(pts1)
# pts2 = np.int32(pts2)
# F, mask = cv.findFundamentalMat(pts1,pts2,cv.FM_LMEDS)
# print(F)
# # We select only inlier points
# pts1 = pts1[mask.ravel()==1]
# pts2 = pts2[mask.ravel()==1]

# def drawlines(img1,img2,lines,pts1,pts2):
#     ''' img1 - image on which we draw the epilines for the points in img2
#         lines - corresponding epilines '''
#     r,c = img1.shape
#     img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
#     img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
#     for r,pt1,pt2 in zip(lines,pts1,pts2):
#         color = tuple(np.random.randint(0,255,3).tolist())
#         x0,y0 = map(int, [0, -r[2]/r[1] ])
#         x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
#         img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
#         img1 = cv.circle(img1,tuple(pt1),5,color,-1)
#         img2 = cv.circle(img2,tuple(pt2),5,color,-1)
#     return img1,img2

# # Find epilines corresponding to points in right image (second image) and
# # drawing its lines on left image
# lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
# lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
# # Find epilines corresponding to points in left image (first image) and
# # drawing its lines on right image
# lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
# lines2 = lines2.reshape(-1,3)
# img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()