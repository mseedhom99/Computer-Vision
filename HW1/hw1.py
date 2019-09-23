import cv2
import math
import numpy as np
import random
import itertools

# Mathew Seedhom
# CS 558
# Homework 1
# I pledge my honor that I have abided by the Stevens Honor System.


def nonMaximumSupression(arr, horizontal, vertical, mode):
    # This function supresses pixel intensities based on the local neighborhood of the pixel, as well as the current use of the function.
    canvas = arr.copy()
    pic = arr.copy()
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if mode == "edges":
                angle = math.atan2(vertical[i][j], horizontal[i][j]) 
                canvas[i][j] = arr[i][j]
                if i == 0 or j == 0 or i == len(arr) - 1  or j == len(arr[i]) - 1:
                    canvas[i][j] = 0
                elif (angle >=  -1*math.pi/8 and angle <= math.pi / 8) or (angle > 7*math.pi/8 and angle <= -7*math.pi/8):
                    if arr[i][j] <= arr[i][j+1] or arr[i][j] <= arr[i][j-1]:
                        canvas[i][j] = 0
                elif (angle < -1*math.pi/8 and angle >= -3*math.pi/8) or (angle > math.pi/8 and angle <= 3*math.pi/8):
                    if arr[i][j] <= arr[i+1][j+1] or arr[i][j] <= arr[i-1][j-1]:
                        canvas[i][j] = 0
                elif (angle < -3*math.pi/8 and angle >= -5*math.pi/8) or (angle > 3*math.pi/8 and angle <= 5*math.pi/8):
                    if arr[i][j] <= arr[i+1][j] or arr[i][j] <= arr[i-1][j]:
                        canvas[i][j] = 0
                elif (angle < -5*math.pi/8 and angle >= -7*math.pi/8) or (angle > 5*math.pi/8 and angle <= 7*math.pi/8):
                    if arr[i][j] <= arr[i+1][j-1] or arr[i][j] <= arr[i-1][j+1]:
                        canvas[i][j] = 0
                else:
                    canvas[i][j] = 0  #hello :)
                if canvas[i][j] < 0:
                    pic[i][j] = 0
                elif canvas[i][j] > 255:
                    pic[i][j] = 255
                else:
                    pic[i][j] = canvas[i][j]
            elif mode == "corners":
                canvas[i][j] = arr[i][j]
                if (i == 0 or j == 0 or i == len(arr) - 1 or j == len(arr[i]) - 1):
                    canvas[i][j] = 0
                elif not (arr[i][j] > arr[i+1][j+1] and arr[i][j] > arr[i-1][j-1] and arr[i][j] > arr[i+1][j-1] and arr[i][j] > arr[i-1][j+1] and arr[i][j] > arr[i][j+1] and arr[i][j] > arr[i][j-1] and arr[i][j] > arr[i+1][j] and arr[i][j] > arr[i-1][j]):
                    canvas[i][j] = 0
                if canvas[i][j] < 0:
                    pic[i][j] = 0
                elif canvas[i][j] > 255:
                    pic[i][j] = 255
                else:
                    pic[i][j] = canvas[i][j]
    return [canvas, pic]

def threshold(canvas, upper, lower):
    # Sets all pixel values below a certain value to 0, while keeping pixel values above the value at a spcific intensity.
    for i in range(len(canvas)):
        for j in range(len(canvas[i])):
            if canvas[i][j] < lower:
                canvas[i][j] = 0
            elif canvas[i][j] < upper:
                canvas[i][j] = 125
            else:
                canvas[i][j] = 255
    return canvas

def connect(arr1, arr2, background = False):
    # Overlays images based on gradient strength
    pic = arr1.copy()
    for i in range(len(arr1)):
        for j in range(len(arr1[i])):
            if not background:
                val = math.sqrt(arr1[i][j] ** 2 + arr2[i][j] ** 2)
                arr1[i][j] = val
                if val < 0:
                    pic[i][j] = 0
                elif val > 255:
                    pic[i][j] = 255
                else:
                    pic[i][j] = val
            else:
                if arr2[i][j] > arr1[i][j]:
                    arr1[i][j] = arr2[i][j]
                if arr1[i][j] < 0:
                    pic[i][j] = 0
                elif arr1[i][j] > 255:
                    pic[i][j] = 255
                else:
                    pic[i][j] = arr1[i][j]
    return (arr1, pic)

def apply_hes(ixx, iyy, ixy, iyx, threshold):
    # Uses the hessian matrix to detect corners.
    arr = ixx.copy()
    pic = ixx.copy()
    for i in range(len(ixx)):
        for j in range(len(ixx[i])):
            det = ixx[i][j]*iyy[i][j]-ixy[i][j]*iyx[i][j]
            trace = ixx[i][j] + iyy[i][j]
            r = det - .06*(trace**2)
            arr[i][j] = r
            if r > threshold:
                pic[i][j] = 255
            else:
                arr[i][j] = 0
                pic[i][j] = 0
    return [arr, pic]

def new_array(pic):
    # Creates a new numpy array for a picture.
    arr = []
    for i in range(len(pic)):
        arr += [[]]
        for j in range(len(pic[i])):
            arr[i] += [pic[i][j]]
    arr = np.array(arr, dtype="float32")
    return arr

def apply_fil(filter, arr):
    # Convolves an image with a given kernel.
    canvas = new_array(arr)
    pic = new_array(arr)
    for i in range(1, len(arr) - 1):
        for j in range(1, len(arr[i]) - 1):
            pix = 0
            for rows in range(len(filter)):
                for columns in range(len(filter[rows])):
                    x = rows - (len(filter) // 2)
                    y = columns - (len(filter[rows]) // 2)
                    pix += filter[rows][columns] * arr[i + x][j + y]
            canvas[i][j] = pix
            if pix < 0:
                pic[i][j] = 0
            elif pix > 255:
                pic[i][j] = 255
            else:
                pic[i][j] = pix
    return (canvas, pic)

def cornersToList(corners):
    # Turns a numpy array of corners into a list.
    listc = []
    for i in range(len(corners)):
        for j in range(len(corners[i])):
            if corners[i][j] > 0:
                listc += [(i, j)]
    return listc

def RANSAC(corners, threshold, inliers, it):
    # Uses the RANSAC algorithm to give structure to a set of points.
    maxpts = []
    passes = []
    success = []
    used = []
    endpoints = []
    for i in range(it):
        maxpts += [[0, 0]]
        passes += [[(0, 0)]]
        success += [0]
        used += [[]]
        endpoints += [[]]
    for j in range(17):
        items = random.sample(range(len(corners)), 2)
        endpoints[j] = [corners[items[0]], corners[items[1]]]
        passes[j] = [corners[items[0]], corners[items[1]]]
        lineD = (corners[items[0]][0] - corners[items[1]][0])**2 + (corners[items[0]][1] - corners[items[1]][1])**2
        try:
            m = (corners[items[0]][0] - corners[items[1]][0]) / (corners[items[0]][1] - corners[items[1]][1])
        except:
            continue
        if m == 0:
            continue
        b = -m*corners[items[0]][1] + corners[items[0]][0]
        for k in range(len(corners)):
            cornerx = (corners[k][1] / m + corners[k][0] - b)/(m + 1/m)
            cornery = cornerx * m + b
            d = (corners[k][1] - cornerx) ** 2 + (corners[k][0] - cornery) ** 2
            if d <= threshold ** 2:
                success[j] += 1
                used[j] += [corners[k]]
                firstD = (corners[k][0] - endpoints[j][0][0])**2 + (corners[k][1] - endpoints[j][0][1])**2
                secondD = (corners[k][0] - endpoints[j][1][0])**2 + (corners[k][1] - endpoints[j][1][1])**2
                if firstD > lineD or secondD > lineD:
                    if firstD <= secondD:
                        if firstD > maxpts[j][0]:
                            maxpts[j][0] = firstD
                            passes[j][0] = corners[k]
                    else:
                        if secondD > maxpts[j][1]:
                            maxpts[j][1] = secondD
                            passes[j][1] = corners[k]
            if success[j] >= inliers:
                return [passes[j], used[j], endpoints[j]]
    winner = success.index(max(success))
    return [passes[winner] , used[winner], endpoints[winner]]

def iterateRANSAC(pic, corners, threshold = math.sqrt(3.84), inliers = 1000, features = 4, it = 17):
    # Applies the RANSAC algorithm to an image a spcified amount of times, and creates an image depicting the result..
    colors = list(itertools.product([0, 255], repeat = 3))
    color = random.sample(colors[1:-1], features + 1)
    for i in range(features):
        winner = RANSAC(corners, threshold, inliers, it)
        pic = cv2.line(pic, (winner[0][0])[::-1], (winner[0][1])[::-1], color[i], 1)
        for j in range(len(winner[1])): 
            point = winner[1][j]
            for row in range(3):
                for column in range(3):
                    x = row - 1
                    y = column - 1
                    if point != winner[2][0] and point != winner[2][1]:
                        pic[point[0] + x][point[1] + y] = color[i]
                    else:
                        pic[point[0] + x][point[1] + y]= color[-1]
            corners.remove(point)
    return pic
        
def houghTransform(pic, corners, angbins, radiusbins, features):
    # Applies the Hough Transform to a set of points.
    topn = []
    for i in range(features):
        topn += [[(0, 0), 0]]
    H = []
    for i in range(radiusbins):
        H += [[]]
        for j in range(angbins):
            H[i] += [0]
    for i in range(len(corners)):
        for j in range(0, angbins):
            r = corners[i][1]*math.cos(j * math.pi / angbins) + corners[i][0]*math.sin(j * math.pi / angbins)
            rbin = math.floor(radiusbins/2 / math.sqrt(len(pic)**2 + len(pic[1])**2) * r + radiusbins/2)
            r = math.floor(r)
            H[rbin][j] += 1
            for k in range(features):
                if H[rbin][j] > topn[k][1]:
                    if [(r, j * math.pi / angbins), H[rbin][j]-1] in topn:
                        index = topn.index([(r, j * math.pi / angbins), H[rbin][j]-1])
                        topn = topn[:k] + [[(r, j * math.pi / angbins), H[rbin][j]]] + topn[k:index] + topn[index+1:]
                    else:
                        topn = topn[:k] + [[(r, j * math.pi / angbins), H[rbin][j]]] + topn[k:-1]
                    break
    return topn

def houghIterator(pic, corners, angbins = 180, radiusbins = None, features = 4):
    # APplies the Hough Transofrm a specifed amount of times, and creates an image depicting the result.
    if radiusbins == None:
        radiusbins = math.ceil(2*math.sqrt(len(pic)**2 + len(pic[1])**2))
    H = houghTransform(pic, corners, angbins, radiusbins, features)
    for i in range(features):
        yf = int(H[i][0][0]/math.sin(H[i][0][1]))
        ys = int((H[i][0][0] - len(pic[1])*math.cos(H[i][0][1]))/math.sin(H[i][0][1]))

        pic = cv2.line(pic, (0, yf), (len(pic[1]), ys), (0, 0 , 255), 1)
    return pic

if __name__ == "__main__":
    gaussian = [[0.077847, 0.123317, 0.077847], 
                [0.123317, 0.195346, 0.123317], 
                [0.077847, 0.123317, 0.077847]]
    sobel_h = [[1, 2, 1], 
               [0, 0, 0], 
               [-1, -2, -1]]
    sobel_v = [[1, 0, -1], 
               [2, 0, -2], 
               [1, 0, -1]]
    print("Working on it!")

    pic = cv2.imread("road.png", 0)
    arr = new_array(pic)

    blurred = apply_fil(gaussian, arr)
    cv2.imwrite("gaussFilter.png", blurred[1])

    h = apply_fil(sobel_h, blurred[0])
    cv2.imwrite("horiEdges.png", h[1])
    v = apply_fil(sobel_v, blurred[0])
    cv2.imwrite("vertEdges.png", v[1])

    edges = connect(h[0], v[0])
    cv2.imwrite("edgesNoSup.png", edges[1])
    edgesSup = nonMaximumSupression(edges[0], h[0], v[0], "edges")
    cv2.imwrite("edgesSup.png", edgesSup[1])
    edgesThresh = threshold(edgesSup[0], 175, 60)
    cv2.imwrite("edgesThresh.png", edgesThresh)

    ixx = apply_fil(sobel_h, h[0])
    iyy = apply_fil(sobel_v, v[0])
    ixy = apply_fil(sobel_v, h[0])
    iyx = apply_fil(sobel_h, v[0])


    cv2.imwrite("Ixx.png", ixx[1])
    cv2.imwrite("Iyy.png", iyy[1])
    cv2.imwrite("Iyx.png", iyx[1])
    cv2.imwrite("Ixy.png", ixy[1])

    hess = apply_hes(ixx[0], iyy[0], ixy[0], iyx[0], 175000)
    hessThresh = nonMaximumSupression(hess[0], h[0], v[0], "corners")
    cv2.imwrite("corners.png", hess[1])
    cv2.imwrite("cornersThresh.png", hessThresh[1])
    newhess = connect(hessThresh[0], edgesThresh / 4, background=True) 
    cv2.imwrite("fullcorners.png", newhess[1])
    fullCOLOR = (newhess[1]).copy()
    fullCOLOR = cv2.cvtColor(newhess[1], cv2.COLOR_GRAY2RGB)
    RANSACimg = fullCOLOR.copy()
    RANSACimg = iterateRANSAC(RANSACimg, cornersToList(hessThresh[1]), it = 25)
    cv2.imwrite("RANSAC.png", RANSACimg)
    HOUGH = fullCOLOR.copy()
    HOUGH = houghIterator(HOUGH, cornersToList(hessThresh[1]), angbins=45)
    cv2.imwrite("HOUGH.png", HOUGH)

    print("All done!")

