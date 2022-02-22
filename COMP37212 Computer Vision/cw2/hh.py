import cv2
import numpy as np
import scipy.ndimage
import scipy.spatial
import math
import copy


def angelCompute(x, y):
    angle = math.atan2(-y, x)
    angle = angle*180/math.pi
    if angle < 0:
        angle = 360.0+angle
    return angle


# 1. Implement python code for Harris Corner detector
def HarrisPointsDetector(img : np.ndarray, sigma=5, threshold=0.1):
    height, width = img.shape[0:2]
    # Calculate the gradients
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Use of reflective Image padding
    # 3. Calculate the 𝐼"! and 𝐼#! partial image derivatives (with respect to x and y) at point p using the sobel operator.
    dx = scipy.ndimage.sobel(img.astype('float'), axis=1, mode="reflect")
    dy = scipy.ndimage.sobel(img.astype('float'), axis=0, mode="reflect")

    # 3. the corresponding Ixx, Iyy, Ixy
    Ixx = dx * dx
    Ixy = dx * dy
    Iyy = dy * dy
    # Get angle for rotation
    # _, ang = cv2.cartToPolar(dx, dy, angleInDegrees=True)
    # Square the derivatives, wp using GaussianMask

    # 4. Find corner response c(M) for every image pixel (using a Gaussian window)
    Ixx = scipy.ndimage.gaussian_filter(Ixx, sigma=0.5, mode="reflect", truncate=9).astype('float')
    Ixy = scipy.ndimage.gaussian_filter(Ixy, sigma=0.5, mode="reflect", truncate=9).astype('float')
    Iyy = scipy.ndimage.gaussian_filter(Iyy, sigma=0.5, mode="reflect", truncate=9).astype('float')
    # sigma = (sigma, sigma)
    # Ixx = cv2.GaussianBlur(dx * dx, sigma, 0.5, borderType=cv2.BORDER_REFLECT)
    # Ixy = cv2.GaussianBlur(dx * dy, sigma, 0.5, borderType=cv2.BORDER_REFLECT)
    # Iyy = cv2.GaussianBlur(dy * dy, sigma, 0.5, borderType=cv2.BORDER_REFLECT)

    # Matrix M
    M = np.array([[Ixx, Ixy], [Ixy, Iyy]])
    # # Calculate the determinate
    Determinant = (M[0, 0] * M[1, 1]) - (M[0, 1] * M[1, 0])
    # # Calculate the trace
    Trace = M[0, 0] + M[1, 1]
    # # Calculate Corner Response function R
    R = Determinant - 0.05 * Trace * Trace

    # 5.Find strong interest points via thresholding and local maxima operations. Plot graph of interest point
    # (keypoint) numbers vs threshold values.(See in Report)
    R_flat = R[:].flatten()
    N = int(len(R_flat) * threshold)
    # Get values in top threshold
    top_k_percentile = np.partition(R_flat, -N)[-N:]
    # Find lowest value in top threshold
    minimum = np.min(top_k_percentile)
    # Set all values less than minimum to 0
    R[R < minimum] = 0
    ang = np.zeros_like(R)
    # Select the strongest interest points (keypoints), which are local maxima in a 3x3 neighbourhood.
    s = 7
    for h in range(R.shape[0] - s):
        for w in range(R.shape[1] - s):
            ang[h][w] = angelCompute(dx[h][w], dy[h][w])
            maximum = np.max(R[h:h + s + 1, w:w + s + 1])
            for i in range(h, h + s + 1):
                for j in range(w, w + s + 1):
                    if R[i, j] != maximum:
                        R[i, j] = 0

    # Return harris corners in [H, W, A, R] format
    features = list(np.where(R > 0))
    features.append(ang[np.where(R > 0)])
    features.append(R[np.where(R > 0)])
    corners = zip(*features)
    keypoints = []
    for h, w, ang, resp in corners:
        # keypoints.append(cv2.KeyPoint(x=float(w), y=float(h), _size=1, _angle=ang, _response=resp))
        keypoints.append(cv2.KeyPoint(x=float(w), y=float(h), _angle=ang, _size=15, _response=resp))
    return keypoints


# 6.Calculate ORB Local features (using ORB descriptor) for your detected interest points.
def featureDescriptor(keypoints, img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=512)
    keypoints, des = orb.compute(img, keypoints)
    return keypoints, des


# 6.Compare with Fast features.(Fast Version)
def opencv_ORB_fast(img1, img2):
    fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True,
                                         type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)  # 获取FAST角点探测器

    kp1 = fast.detect(img1, None)
    kp2 = fast.detect(img2, None)
    orb = cv2.ORB_create()

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # kp = orb.detect(img, None)

    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)
    # img1 = cv2.drawKeypoints(img1, kp1, img1, color=(0, 255, 0), flags=0)
    # cv2.imshow('p', img1)
    # cv2.waitKey()
    # img2 = cv2.drawKeypoints(img2, kp2, img2, color=(0, 255, 0), flags=0)
    # cv2.imshow('p', img2)
    # cv2.waitKey()
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # matches = bf.match(des1, des2)
    # print(matches)
    matches = RatioFeatureMatcher(des1, des2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], img2, flags=2)
    # cv2.imshow('p', img3)
    # cv2.waitKey()
    cv2.imwrite('output/fast_' + imgpath, img3)
    return img3


# 6.Compare with build-in ORB features.(Fast Version)
def opencv_ORB(img1, img2):
    orb = cv2.ORB_create()
    kp1 = orb.detect(img, None)
    kp2 = orb.detect(img2, None)

    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)
    # img1 = cv2.drawKeypoints(img1, kp1, img1, color=(0, 255, 0), flags=0)
    # cv2.imshow('p', img1)
    # cv2.waitKey()
    # img2 = cv2.drawKeypoints(img2, kp2, img2, color=(0, 255, 0), flags=0)
    # cv2.imshow('p', img2)
    # cv2.waitKey()
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # matches = bf.match(des1, des2)
    # matches = RatioFeatureMatcher(des1, des2)
    # print(matches)
    matches = RatioFeatureMatcher(des1, des2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], img2, flags=2)
    # cv2.imshow('p', img3)
    # cv2.waitKey()
    cv2.imwrite('output/matcher_builtin_' + imgpath, img3)
    return img3


# 7.Implement sum of squared distances to measure Local Feature similarity SSD.
def SSDFeatureMatcher(des1, des2, featureNum=100):
    distance = []
    numPoint1 = len(des1)
    numPoint2 = len(des2)
    featureNum = min(min(numPoint1, numPoint2), featureNum)
    dist = scipy.spatial.distance.cdist(des1, des2, metric='euclidean')
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


# 7.Implement ratio test to discard points that will give ambiguous matches.
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

# Test My Matcher
def test_ssd(img1, img2):
    orb = cv2.ORB_create()
    kp1 = orb.detect(img, None)
    kp2 = orb.detect(img2, None)

    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)
    img1 = cv2.drawKeypoints(img1, kp1, img1, color=(0, 255, 0), flags=0)
    cv2.imshow('p', img1)
    cv2.waitKey()
    img2 = cv2.drawKeypoints(img2, kp2, img2, color=(0, 255, 0), flags=0)
    cv2.imshow('p', img2)
    cv2.waitKey()
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # matches = bf.match(des1, des2)
    # matches = SSDFeatureMatcher(des1, des2)
    matches = RatioFeatureMatcher(des1, des2)
    # print(matches)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[0:10], img2, flags=2)
    cv2.imshow('p', img3)
    cv2.waitKey()


# Test My Harris
def test_harris(img1, img2):
    orb = cv2.ORB_create()
    # kp1 = orb.detect(img, None)
    # kp2 = orb.detect(img2, None)


    harrisP1 = HarrisPointsDetector(img,threshold=0.001)
    harrisP2 = HarrisPointsDetector(img2,threshold=0.001)
    kp1, des1 = orb.compute(img1, harrisP1)
    kp2, des2 = orb.compute(img2, harrisP2)
    # kp1, des1 = featureDescriptor(harrisP1, img)
    # kp2, des2 = featureDescriptor(harrisP2, img2)
    img1 = cv2.drawKeypoints(img1, kp1, img1, color=(0, 255, 0), flags=0)
    cv2.imshow('p', img1)
    cv2.waitKey()
    img2 = cv2.drawKeypoints(img2, kp2, img2, color=(0, 255, 0), flags=0)
    cv2.imshow('p', img2)
    cv2.waitKey()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)
    # matches = SSDFeatureMatcher(des1, des2)
    # print(matches)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[0:10], img2, flags=2)
    cv2.imshow('p', img3)
    cv2.waitKey()


# Test whole pipeline
def test_pipeline(img1, img2):
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
    kp = HarrisPointsDetector(img)
    cv2.drawKeypoints(img,kp,img, color=(255,0,0))
    img2 = cv2.imread('otherimages/bernieNoisy2.png')
    img2 = cv2.resize(img2, None, fx=img.shape[1] / img2.shape[1], fy=img.shape[1] / img2.shape[1])
    # test_matcher(img, img2)
    # test_harris(img, img2)
    # test_pipeline(img, img2)
    import os
    imgs = os.listdir('otherimages')
    for imgpath in imgs:
        img2 = cv2.imread(os.path.join('otherimages', imgpath))
        img2 = cv2.resize(img2, None, fx=img.shape[1] / img2.shape[1], fy=img.shape[1] / img2.shape[1])
        # self harris pipeline
        img3 = test_pipeline(img, img2)
        # fast offical
        # img3 = opencv_ORB_fast(img, img2)
        # ORB builtin
        # img3 = opencv_ORB(img, img2)