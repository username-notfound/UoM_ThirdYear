import cv2
import numpy as np
import scipy.ndimage
import scipy.spatial
import math
import copy
from matplotlib import pyplot as plt
import os

def angelCompute(x, y):
    angle = (math.atan2(-y, x))*180/math.pi
    if angle < 0:
        angle = 360.0+angle
    return angle

# ---------------------------
# Task 1: Feature detection
def HarrisPointsDetector(img : np.ndarray, sigma=0.5, threshold=0.1):
    height, width = img.shape[0:2]
    # Calculate the gradients
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use the 3*3 Sobel operator to compute the x,y derivatives
    # for pixels outside the image, pad the image using reflection
    dx = scipy.ndimage.sobel(img.astype('float'), axis=1, mode="reflect")
    dy = scipy.ndimage.sobel(img.astype('float'), axis=0, mode="reflect")

    # Get Ix^2, Iy^2, IxIy
    Ixx = dx * dx
    Iyy = dy * dy
    IxIy = dx * dy

    # use a 5x5 Gaussian mask with 0.5 sigma
    Ixx = scipy.ndimage.gaussian_filter(Ixx, sigma=0.5).astype('float')
    IxIy = scipy.ndimage.gaussian_filter(IxIy, sigma=0.5).astype('float')
    Iyy = scipy.ndimage.gaussian_filter(Iyy, sigma=0.5).astype('float')

    # Harris Matrix M
    M = np.array([[Ixx, IxIy], [IxIy, Iyy]])
    detM = Ixx * Iyy - IxIy ** 2
    traceM = Ixx + Iyy
    # Corner strength function, c(M)
    cM = detM - 0.1 * traceM * traceM 

    # Find strong interest points via thresholding and local maxima operations. 
    oneD_cM = cM[:].flatten()
    N = int(len(oneD_cM) * threshold)
    low_thres = np.min(np.partition(oneD_cM, -N)[-N:])
    cM[cM < low_thres] = 0
    ang = np.zeros_like(cM)
    # Select the strongest interest points (keypoints)
    # which are local maxima in a 7x7 neighbourhood.
    kernel = 7
    for h in range(cM.shape[0] - kernel):
        for w in range(cM.shape[1] - kernel):
            ang[h][w] = angelCompute(dx[h][w], dy[h][w])
            maximum = np.max(cM[h:h + kernel + 1, w:w + kernel + 1])
            for i in range(h, h + kernel + 1):
                for j in range(w, w + kernel + 1):
                    if cM[i, j] != maximum:
                        cM[i, j] = 0

    # Return harris corners in [H, W, A, R] format
    features = list(np.where(cM > 0))
    features.append(ang[np.where(cM > 0)])
    features.append(cM[np.where(cM > 0)])
    corners = zip(*features)
    keypoints = []
    for h, w, ang, resp in corners:
        keypoints.append(cv2.KeyPoint(x=float(w), y=float(h), _angle=ang, _size=15, _response=resp))

    return keypoints


# ---------------------------
# Task 2: Feature description
def featureDescriptor(keypoints, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=512)
    keypoints, des = orb.compute(img, keypoints)
    return keypoints, des


# Compare with Fast features
def opencv_ORB_fast(img1, img2):
    fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True,
                                         type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

    kp1 = fast.detect(img1, None)
    kp2 = fast.detect(img2, None)
    orb = cv2.ORB_create()

    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)

    matches = RatioFeatureMatcher(des1, des2)
    result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], img2, flags=2)
    # cv2.imshow('p', img3)
    # cv2.waitKey()
    cv2.imwrite('output/fastORB_' + imgpath, result)
    return result


# Compare with build-in ORB features
def opencv_ORB(img1, img2):
    orb = cv2.ORB_create()
    kp1 = orb.detect(img, None)
    kp2 = orb.detect(img2, None)
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)
    matches = RatioFeatureMatcher(des1, des2)
    res = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], img2, flags=2)
    cv2.imwrite('output/builtin_' + imgpath, res)
    return res

# ---------------------------
# Task 3: Feature matching
# Sum of squared differences (SSD): the squared Euclidean distance between the two feature vectors.
def SSDFeatureMatcher(fv1, fv2, featureNum=100):
    distance = []
    numPoint1 = len(fv1)
    numPoint2 = len(fv2)
    featureNum = min(min(numPoint1, numPoint2), featureNum)
    dist = scipy.spatial.distance.cdist(fv1, fv2, metric='euclidean')
    for i in range(numPoint1):
        for j in range(numPoint2):
            distance.append([i, j, dist[i][j]])
    distance.sort(key=lambda x:x[2])
    # print(distance)
    result = []
    for i in range(featureNum):
        dm = cv2.DMatch(_distance=distance[i][2], _trainIdx=distance[i][1], _queryIdx=distance[i][0], _imgIdx=0)
        result.append(dm)
    return result


# The ratio test: Find the closest and second closest features by SSD distance.
def RatioFeatureMatcher(des1, des2, featureNum=100, threshold=0.7):
    distance = []
    numPoint1 = len(des1)
    numPoint2 = len(des2)
    featureNum = min(min(numPoint1, numPoint2), featureNum)
    dist = scipy.spatial.distance.cdist(des1, des2, metric='euclidean')
    for i in range(numPoint1):
        n1 = np.argmin(dist[i])
        d1 = dist[i][n1]
        dist[i][n1] = 100000
        n2 = np.argmin(dist[i])
        d2 = dist[i][n2]
        ratio = d1/d2
        if ratio < threshold:
            distance.append([i, n1, d1])
    distance.sort(key=lambda x: x[2])
    # print(distance)
    result = []
    for i in range(min(len(distance),featureNum)):
        dm = cv2.DMatch(_distance=distance[i][2], _trainIdx=distance[i][1], _queryIdx=distance[i][0], _imgIdx=0)
        result.append(dm)
    return result

# Output the images
def matching(img1, img2):
    harrisP1 = HarrisPointsDetector(img1, threshold=0.001)
    harrisP2 = HarrisPointsDetector(img2, threshold=0.001)
    kp1, des1 = featureDescriptor(harrisP1, img)
    kp2, des2 = featureDescriptor(harrisP2, img2)
    # matches = SSDFeatureMatcher(des1, des2)
    matches = RatioFeatureMatcher(des1, des2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[0:10], img2, flags=2)
    # cv2.imshow('p', img3)
    # cv2.waitKey()
    cv2.imwrite('output/matcher_' + imgpath, img3)
    return img3


if __name__=='__main__':
    img = cv2.imread('bernieSanders.jpg')
    img = cv2.resize(img, None, fx=0.5, fy=0.5)

    # # -------- Plot the graph --------
    # thres = [0.0001, 0.001, 0.01, 0.1]
    # num_of_kp = []
    # for i in range (0, len(thres)):
    #     imgcopy = img.copy()
    #     kp, num = HarrisPointsDetector(img, 0.5, thres[i])
    #     num_of_kp.append(num)
    #     imgcopy = img.copy()
    #     imgcopy = cv2.drawKeypoints(imgcopy,kp,imgcopy, color=(255,0,0))
    #     window_name = 'threshold = '+ str(thres[i])
    #     cv2.imshow(window_name, imgcopy)
    #     cv2.waitKey(10)
    #     cv2.destroyWindow(window_name)
    # #     imgcopy = cv2.drawKeypoints(imgcopy,kp, None, color=(0,255,0))
    # #     window_name = 'image'
    # #     # cv2.imshow("image", imgcopy)
    # #     # cv2.waitKey(0)
    # #     plt.subplot(2,2,i+1),plt.imshow(imgcopy)
    # #     plt.title(thres[i])

    # #  Plot the number of detected feature points as you vary the threshold value
    # plt.plot(thres, num_of_kp)
    # plt.plot(thres, num_of_kp, 'ro')
    # plt.xlabel('threshold')
    # plt.ylabel('number of detected feature points')
    # plt.title('The number of detected feature points corresponding to the threshold value')
    # plt.show()
    # # -------- END of Plot the graph --------

    img2 = cv2.imread('benchmark/bernieNoisy2.png')
    img2 = cv2.resize(img2, None, fx=img.shape[1] / img2.shape[1], fy=img.shape[1] / img2.shape[1])
    testing_img = os.listdir('benchmark')
    # for imgpath in testing_img:
         img2 = cv2.imread(os.path.join('benchmark', imgpath))
         img2 = cv2.resize(img2, None, fx=img.shape[1] / img2.shape[1], fy=img.shape[1] / img2.shape[1])
         img3 = matching(img, img2)
    #     # fast offical
    #     #img4 = opencv_ORB_fast(img, img2)
    #     # ORB builtin
    #     #img5 = opencv_ORB(img, img2)
    