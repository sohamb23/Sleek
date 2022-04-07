import math
import numpy as np
import statistics
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    ocrTexts = []
    ocrVertices = []
    #print('Texts:')

    for text in texts:
        ocrText = "{}".format(text.description)
        ocrTexts.append(ocrText)
        #print('\n"{}"'.format(text.description))


        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])
        
        #print(vertex.x for vertex in text.bounding_poly.vertices)
        ocrVertices.append(vertices)

        #print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return ocrTexts,ocrVertices

#imageText,imageVertices = detect_text('/Users/sohambose/Harvard/Sleek/MenuDataset/simpleImages/menuImage5.jpeg')
#print(imageVertices)
#print(imageText)
# for i in range(1,len(imageText)):
#     print(imageText[i] + imageVertices[i][-1], end="\t")
#print(x)
#print(x[0])
i = 0

def vertexArrCreator(detectedVertices):
    allVertices = []
    for boxVertices in detectedVertices:
        actualVertices = []
        currStr = ""
        for vertexPair in boxVertices:
            #print(thing)
            vertexPairArr = []
            for i in range(1,len(vertexPair)):
                if(vertexPair[i] != "," and vertexPair[i] != ")"):
                    #print(vertexPair[i])
                    currStr += vertexPair[i]
                else:
                    #print(currStr)
                    vertexPairArr.append(int(currStr))
                    currStr = ""
            actualVertices.append(vertexPairArr)
        allVertices.append(actualVertices)

    return allVertices
# integerImageVertices = vertexArrCreator(imageVertices)
# print(len(integerImageVertices) == len(imageVertices))

def clockwiseRotation(origin, point, angle):
    xOrigin, yOrigin = origin[0], origin[1]
    xPoint, yPoint = point[0], point[1]

    xNew = int(xOrigin + math.cos(-angle) * (xPoint - xOrigin) - math.sin(-angle) * (yPoint - yOrigin))
    yNew = int(yOrigin + math.sin(-angle) * (xPoint - xOrigin) + math.cos(-angle) * (yPoint - yOrigin))
    return [xNew, yNew]

def boxCoordinateRotation(integerVertices):
    slopeArray = []
    for boxVertices in integerVertices:
        if(boxVertices[0][0] - boxVertices[1][0] != 0):
            boxSlope = (boxVertices[0][1] - boxVertices[1][1])/(boxVertices[0][0] - boxVertices[1][0])
            slopeArray.append(boxSlope)
        else:
            slopeArray.append(0)
    slopeMode = statistics.mode(slopeArray)
    #print(statistics.median(slopeArray))
    rotationAngle = np.degrees(np.arctan(slopeMode))
    #print(slopeMode)
    #print(rotationAngle)
    return rotationAngle

def uncroppedRotation(imgPath, angle):
    rotateImage = cv2.imread(imgPath)
    imgHeight, imgWidth = rotateImage.shape[0], rotateImage.shape[1]
 
    centreY, centreX = imgHeight//2, imgWidth//2
 
    rotationMatrix = cv2.getRotationMatrix2D((centreY, centreX), angle, 1.0)
 
    cosofRotationMatrix = np.abs(rotationMatrix[0][0])
    sinofRotationMatrix = np.abs(rotationMatrix[0][1])
 
    newImageHeight = int((imgHeight * sinofRotationMatrix) + (imgWidth * cosofRotationMatrix))
    newImageWidth = int((imgHeight * cosofRotationMatrix) + (imgWidth * sinofRotationMatrix))
 
    rotationMatrix[0][2] += (newImageWidth/2) - centreX
    rotationMatrix[1][2] += (newImageHeight/2) - centreY
 
    rotatingimage = cv2.warpAffine(rotateImage, rotationMatrix, (newImageWidth, newImageHeight))
    if(angle != 0):
        cv2.imshow('img',rotatingimage)
        cv2.waitKey(0)
        output_path = "/Users/sohambose/Harvard/Sleek/MenuDataset/deskewedImages"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        file_name = os.path.basename(imgPath).split('.')[0]
        file_name = file_name.split()[0]
        save_path = os.path.join(output_path, file_name + "_deskewed_" + ".jpg")
        cv2.imwrite(save_path, rotatingimage)
        print(save_path)
 
    return rotatingimage,save_path

def rotateImage(imgPath, angle):
    image = cv2.imread(imgPath)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    if(angle > 0):
        cv2.imshow('img',result)
        cv2.waitKey(0)
        output_path = "/Users/sohambose/Harvard/Sleek/MenuDataset/deskewedImages"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        file_name = os.path.basename(imgPath).split('.')[0]
        file_name = file_name.split()[0]
        save_path = os.path.join(output_path, file_name + "_deskewed_" + ".jpg")
        cv2.imwrite(save_path, result)
        #print(save_path)
    return result

def clusterThreshold(wordText, wordVertices):
    integerVertices = vertexArrCreator(wordVertices)
    boxHeightArr = []
    for i in range(1, len(integerVertices)):
        boxHeight = integerVertices[i][-2][-1] - integerVertices[i][-3][-1]
        boxHeightArr.append(boxHeight)
    medianBoxHeight = statistics.median(boxHeightArr)



def geometricClustering(wordText, wordVertices):
    #cluster through while loop with x and y thresholding
    integerVertices = vertexArrCreator(wordVertices)
    boxHeightArr = []
    for i in range(1, len(integerVertices)):
        boxHeight = integerVertices[i][-2][-1] - integerVertices[i][-3][-1]
        boxHeightArr.append(boxHeight)
    medianBoxHeight = statistics.median(boxHeightArr)
    #print(medianBoxHeight)

    lpcArr = []                                     #length per character
    for i in range(1,len(integerVertices) - 1):
        xcurrTopRight = integerVertices[i][-2][0]
        xcurrTopLeft = integerVertices[i][-1][0]
        boxLength = abs(xcurrTopRight - xcurrTopLeft)
        numChars = len(wordText[i])
        lpc = boxLength/numChars
        lpcArr.append(lpc)
    #print(lpcArr)
    #print(statistics.median(lpcArr))

    whitespaceArr = []
    for i in range(1,len(integerVertices) - 1):
        xcurrTopRight = integerVertices[i][-2][-2]
        xnextTopLeft = integerVertices[i+1][-1][-2]
        whitespace = abs(xnextTopLeft - xcurrTopRight)
        whitespaceArr.append(whitespace)
    whitespaceMode = statistics.mode(whitespaceArr)
    #print(whitespaceMode)
    #print(whitespaceArr)


    clusterArray = []
    currCluster = [wordText[1]]
    for i in range(1, len(integerVertices)-1):
        ycurrTopRight = integerVertices[i][-2][-1]
        ycurrBottomRight = integerVertices[i][1][-1]
        ynextTopRight = integerVertices[i+1][-2][-1]
        ynextBottomRight = integerVertices[i+1][1][-1]
        xcurrTopRight = integerVertices[i][-2][-2]
        xnextTopLeft = integerVertices[i+1][-1][-2]
        #print(xcurrTopRight-xnextTopLeft)
        if((((ynextTopRight < ycurrTopRight+5) and (ynextTopRight > ycurrBottomRight-5)) or ((ynextBottomRight < ycurrTopRight + 5) and (ynextBottomRight > ycurrBottomRight - 5))) and (abs(xnextTopLeft-xcurrTopRight) < 15)): #change the value of 5 to the median box height 
            currCluster.append(wordText[i+1])
        else:
            if currCluster:
                clusterArray.append(currCluster)
            currCluster = [wordText[i+1]]
    #print(clusterArray)
    clusterArray.append(currCluster)
    return clusterArray

    # for i in range(1,len(integerVertices)):
    #     print(wordText[i], end = " ")
    #     print(integerVertices[i][-2][-1] - integerVertices[i][-3][-1])

def priceClassification(clusterArray, wordText, wordVertices):
    unclusteredArray = []
    for i in range(len(clusterArray)):
        for j in range(len(clusterArray[i])):
            unclusteredArray.append(clusterArray[i][j])

    integerVertices = vertexArrCreator(wordVertices) #turn string vertices into integers

    boxHeightArr = []
    for i in range(1, len(integerVertices)):
        boxHeight = integerVertices[i][-2][-1] - integerVertices[i][-3][-1]
        boxHeightArr.append(boxHeight)
    medianBoxHeight = statistics.median(boxHeightArr)

    wordCounter = 0
    clusterTopLefts = []
    #clusterTopLeftsAverage = []
    clusterHeights = []
    clusterHeightsAverage = []
    for i in range(len(clusterArray)):
        wordTopLefts = []
        wordHeights = []
        for j in range(len(clusterArray[i])):
            wordCounter += 1
            wordTopLeft = integerVertices[wordCounter][-1]
            wordTopLefts.append(wordTopLeft)
            wordHeight = integerVertices[wordCounter][-2][-1] - integerVertices[wordCounter][-3][-1]
            wordHeights.append(wordHeight)

        clusterTopLefts.append(wordTopLefts)
        #clusterTopLeftAverage = sum(wordTopLefts)/len(wordTopLefts)
        #clusterTopLeftsAverage.append(clusterTopLeftAverage)
        #print(clusterTopLefts)
        clusterHeights.append(wordHeights)
        clusterHeightAverage = sum(wordHeights)/len(wordHeights)
        clusterHeightsAverage.append(clusterHeightAverage)
    
    indexCounter = 1
    clusterAPC = []                             #cluster area per character
    
    for i in range(len(clusterArray)):
        charCounter = 0
        firstBottomLeft = integerVertices[indexCounter][0]
        lastTopRight = integerVertices[indexCounter + len(clusterArray[i]) - 1][-2]
        boxArea = abs(lastTopRight[0] - firstBottomLeft[0]) * abs(lastTopRight[1] - firstBottomLeft[0])
        for j in range(len(clusterArray[i])):
            for char in clusterArray[i][j]:
                charCounter += 1

        for k in range(len(clusterArray[i])):
            indexCounter += 1
        valClusterAPC = boxArea/charCounter
        clusterAPC.append(valClusterAPC)
    #print(clusterAPC)



    clusterDict = {}
    wordCounterTwo = 0
    kArr = []
    for i in range(len(clusterArray)):
        clusterDict[i] = None
        digitCounter = 0
        alphaCounter = 0
        lowerCounter = 0
        upperCounter = 0
        charCounter = 0
        for j in range(len(clusterArray[i])):
            #clusterHeightAverage = sum(clusterArray[i])/len(clusterArray[i])
            for char in clusterArray[i][j]:
                if(char.isdigit()):
                    digitCounter += 1
                if(char.isalpha()):
                    alphaCounter += 1
                if(char.islower()):
                    lowerCounter += 1
                if(char.isupper()):
                    upperCounter += 1
                charCounter += 1
        # if(clusterDict[i] == None):
        #     #print(clusterArray[i])
        #     #print("Menu Item\n")
        #     clusterDict[i] = "Menu Item"
        if((digitCounter/charCounter) > 0.5 and charCounter < 10):
            #print(clusterArray[i])
            #print("Price\n")
            clusterDict[i] = "Price"
    return clusterDict



def overallClassification(imgPath, clusterArray, wordText, wordVertices, clusterDict, distanceThresholds, overallFontGroup):
    img = cv2.imread(imgPath)
    height = img.shape[0]
    width = img.shape[1]

    priceClusterDict = clusterDict

    unclusteredArray = []
    for i in range(len(clusterArray)):
        for j in range(len(clusterArray[i])):
            unclusteredArray.append(clusterArray[i][j])

    integerVertices = vertexArrCreator(wordVertices) #turn string vertices into integers

    boxHeightArr = []
    for i in range(1, len(integerVertices)):
        boxHeight = integerVertices[i][-2][-1] - integerVertices[i][-3][-1]
        boxHeightArr.append(boxHeight)
    medianBoxHeight = statistics.median(boxHeightArr)

    wordCounter = 0
    clusterTopLefts = []
    clusterBottomLefts = []
    #clusterTopLeftsAverage = []
    clusterHeights = []
    clusterHeightsAverage = []
    for i in range(len(clusterArray)):
        wordTopLefts = []
        wordBottomLefts = []
        wordHeights = []
        for j in range(len(clusterArray[i])):
            wordCounter += 1
            wordTopLeft = integerVertices[wordCounter][-1]
            wordBottomLeft = integerVertices[wordCounter][0]
            wordTopLefts.append(wordTopLeft)
            wordBottomLefts.append(wordBottomLeft)
            wordHeight = integerVertices[wordCounter][-2][-1] - integerVertices[wordCounter][-3][-1]
            wordHeights.append(wordHeight)

        clusterTopLefts.append(wordTopLefts)
        clusterBottomLefts.append(wordBottomLefts)
        #clusterTopLeftAverage = sum(wordTopLefts)/len(wordTopLefts)
        #clusterTopLeftsAverage.append(clusterTopLeftAverage)
        #print(clusterTopLefts)
        clusterHeights.append(wordHeights)
        clusterHeightAverage = sum(wordHeights)/len(wordHeights)
        clusterHeightsAverage.append(clusterHeightAverage)
    
    # indexCounter = 1
    # allWordAPC = []                             #cluster area per character
    
    # for i in range(len(clusterArray)):
    #     wordClusterAPC = []
    #     charCounter = 0
    #     #firstBottomLeft = integerVertices[indexCounter][0]
    #     #lastTopRight = integerVertices[indexCounter + len(clusterArray[i]) - 1][-2]
    #     for j in range(len(clusterArray[i])):
    #         wordBottomLeft = integerVertices[indexCounter][0]
    #         #print(wordBottomLeft)
    #         wordTopRight = integerVertices[indexCounter][-2]
    #         wordArea = (abs(wordTopRight[0] - wordBottomLeft[0])) * (abs(wordTopRight[1] - wordBottomLeft[1]))
    #         #print(wordArea)
    #         for char in clusterArray[i][j]:
    #             charCounter += 1
    #         indexCounter += 1
    #         valWordAPC = wordArea/charCounter
    #         wordClusterAPC.append(valWordAPC)
    #     allWordAPC.append(wordClusterAPC)
    #print(integerVertices)
    #print(allWordAPC)
        
        
    #print(clusterAPC)



    #clusterDict = {}
    indexCounter = 1
    clusterInitIndex = []
    for i in range(len(clusterArray)):
        clusterInitIndex.append(indexCounter)
        for j in range(len(clusterArray[i])):
            indexCounter += 1

    allClusterLineHeightArr = []
    for i in range(len(clusterInitIndex) - 1):
        minClusterLineHeightArr = []
        minClusterLineHeight = 1000
        for j in range(i + 1,len(clusterInitIndex)):
            #print(abs(integerVertices[clusterInitIndex[i]][-1][0] - integerVertices[clusterInitIndex[j]][-1][0]))
            if((clusterDict[i] != "Price" and clusterDict[j] != "Price" and (abs(integerVertices[clusterInitIndex[i]][-1][0] - integerVertices[clusterInitIndex[j]][-1][0])<0.4*width))): #change 200 to 1/2 of picture image length
                clusterLineHeight = abs(integerVertices[clusterInitIndex[i]][0][1] - integerVertices[clusterInitIndex[j]][-1][1])
                #print(clusterLineHeight)
                if(clusterLineHeight < minClusterLineHeight):
                    minClusterLineHeight = clusterLineHeight
                    allClusterLineHeightArr.append([minClusterLineHeight,i,j])
                # for k in range(len(distanceThresholds)):
                #     if(minClusterLineHeight > distanceThresholds[k] - 4 and minClusterLineHeight < distanceThresholds[k] + 4):
                #         if(k == 0):
                #             clusterDict[i] = "Menu Item"
                #             clusterDict[j] = "Menu Description"
                #         if(k == 1):
                #             clusterDict[i] = "Menu Description"
                #             clusterDict[j] = "Menu Item"
                #         if(k == 2):
                #             clusterDict[i] = "Category"
                #             clusterDict[j] = "Menu Item"

    overallFontGroup, medianClusterAPC = findWordAPC(clusterArray, wordVertices)
    
    for i in range(len(allClusterLineHeightArr)):
        lineHeight = allClusterLineHeightArr[i][0]
        if(len(distanceThresholds) == 2):
            if(lineHeight < distanceThresholds[0]):
                clusterDict[allClusterLineHeightArr[i][2]] = "Menu Item"
                #clusterDict[allClusterLineHeightArr[i][2]] = None
                #clusterDict[allClusterLineHeightArr[i][2]] = "Menu Description"
            elif(lineHeight < distanceThresholds[1] and lineHeight > distanceThresholds[0]):
                clusterDict[allClusterLineHeightArr[i][2]] = "Menu Item"
            else:
                clusterDict[allClusterLineHeightArr[i][2]] = "Menu Item"
        else:
            if(lineHeight < distanceThresholds[0]):
                #clusterDict[allClusterLineHeightArr[i][2]] = None
                clusterDict[allClusterLineHeightArr[i][2]] = "Menu Description"
            elif(lineHeight < distanceThresholds[1] and lineHeight > distanceThresholds[0]):
                clusterDict[allClusterLineHeightArr[i][2]] = "Menu Item"
            else:

                clusterDict[allClusterLineHeightArr[i][2]] = "Menu Item"

                

            # if(lineHeight > distanceThresholds[k] - 4 and lineHeight < distanceThresholds[k] + 4):
            #     if(k == 0):
            #         print(clusterArray[allClusterLineHeightArr[i][1]])
            #         print(clusterArray[allClusterLineHeightArr[i][2]])
            #         print("Menu Description")
            #         print("\n")
            #         # if((medianClusterAPC[allClusterLineHeightArr[i][1]] < 1.5 * medianClusterAPC[allClusterLineHeightArr[i][2]]) and (medianClusterAPC[allClusterLineHeightArr[i][1]] > 0.83 * medianClusterAPC[allClusterLineHeightArr[i][2]])):
            #         #     clusterDict[allClusterLineHeightArr[i][1]] = "Menu Item"
            #         #     clusterDict[allClusterLineHeightArr[i][2]] = "Menu Item"
            #         #     print("MIMI")
            #         # else:
            #         #     clusterDict[allClusterLineHeightArr[i][1]] = "Menu Item"
            #         #     clusterDict[allClusterLineHeightArr[i][2]] = "Menu Description"
            #         #     print("MIMD")
            #         clusterDict[allClusterLineHeightArr[i][2]] = "Menu Description"
            #     if(k == 1):
            #         print(clusterArray[allClusterLineHeightArr[i][1]])
            #         print(clusterArray[allClusterLineHeightArr[i][2]])
            #         print("Menu Item")
            #         print("\n")
            #         # if((medianClusterAPC[allClusterLineHeightArr[i][1]] < 1.5 * medianClusterAPC[allClusterLineHeightArr[i][2]]) and (medianClusterAPC[allClusterLineHeightArr[i][1]] > 0.83 * medianClusterAPC[allClusterLineHeightArr[i][2]])):
            #         #     clusterDict[allClusterLineHeightArr[i][1]] = "Menu Item"
            #         #     clusterDict[allClusterLineHeightArr[i][2]] = "Menu Item"
            #         #     print("MIMI")
            #         # else:
            #         #     clusterDict[allClusterLineHeightArr[i][1]] = "Menu Description"
            #         #     clusterDict[allClusterLineHeightArr[i][2]] = "Menu Item"
            #         #     print("MDMI")
            #         clusterDict[allClusterLineHeightArr[i][2]] = "Menu Item"

            #     if(k == 2):
            #         print(clusterArray[allClusterLineHeightArr[i][1]])
            #         print(clusterArray[allClusterLineHeightArr[i][2]])
            #         print("Menu Item")
            #         print("\n")
            #         # print("CMI")
            #         # clusterDict[allClusterLineHeightArr[i][1]] = "Category"
            #         # clusterDict[allClusterLineHeightArr[i][2]] = "Menu Item"
            #         clusterDict[allClusterLineHeightArr[i][2]] = "Menu Item"

    
    # for i in range(len(medianClusterAPC)):
    #     print(clusterArray[i])
    #     print(medianClusterAPC[i])

    for i in range(len(clusterArray)):
        for j in range(len(clusterArray[i])):
            if(clusterHeightsAverage[i] > (1.5)*(medianBoxHeight)):
                #print(clusterArray[i])
                #print("Category\n")
                clusterDict[i] = "Category"

    # categoryComparator = overallFontGroup[-1][0]
    # for i in range(len(medianClusterAPC)):
    #     if(medianClusterAPC[i] <= categoryComparator * 1.3 and medianClusterAPC[i] >= categoryComparator):
    #         clusterDict[i] = "Category"
            


    wordCounterTwo = 0
    kArr = []
    for i in range(len(clusterArray)):
        #clusterDict[i] = None
        digitCounter = 0
        alphaCounter = 0
        lowerCounter = 0
        upperCounter = 0
        charCounter = 0
        for j in range(len(clusterArray[i])):
            #clusterHeightAverage = sum(clusterArray[i])/len(clusterArray[i])
            for char in clusterArray[i][j]:
                if(char.isdigit()):
                    digitCounter += 1
                if(char.isalpha()):
                    alphaCounter += 1
                if(char.islower()):
                    lowerCounter += 1
                if(char.isupper()):
                    upperCounter += 1
                charCounter += 1
        # if(clusterDict[i] == None):
        #     #print(clusterArray[i])
        #     #print("Menu Item\n")
        #     clusterDict[i] = "Menu Item"
        # if((digitCounter/charCounter) > 0.5 and charCounter < 10):
        #     #print(clusterArray[i])
        #     #print("Price\n")
        #     clusterDict[i] = "Price"
        # elif(clusterHeightsAverage[i] > (1.5)*(medianBoxHeight)):
        #     #print(clusterArray[i])
        #     #print("Category\n")
        #     clusterDict[i] = "Category"
        
    #     if(i < len(clusterArray) - 5):
    #         for k in range(i+1, i + 5):
    #             if(clusterDict[i] != "Price" and clusterDict[k] != "Price"):
    #                 if((abs(clusterBottomLefts[i][0][1] - clusterTopLefts[k][0][1]) < distanceThresholds[0] + 5) and (abs(clusterBottomLefts[i][0][1] - clusterTopLefts[k][0][1]) > distanceThresholds[0] - 5)):
    #                     #print(clusterArray[i])
    #                     #print(clusterArray[k])
    #                     clusterDict[i] = "Menu Item"
    #                     #print(k)
    #                     kArr.append(int(k))
    #                     #print(clusterArray[k])
    #                     #print("Menu Description\n")
    #                     kArr.append(k)
    #                     clusterDict[k] = "Menu Description"
    #                     #print(k)
    #                     #print(clusterArray[k])
    #                 if((abs(clusterBottomLefts[i][0][1] - clusterTopLefts[k][0][1]) < distanceThresholds[1] + 5) and (abs(clusterBottomLefts[i][0][1] - clusterTopLefts[k][0][1]) > distanceThresholds[1] - 5)):
    #                     clusterDict[i] = "Menu Description"
    #                     clusterDict[k] = "Menu Item"
    #                 if((abs(clusterBottomLefts[i][0][1] - clusterTopLefts[k][0][1]) < distanceThresholds[2] + 5) and (abs(clusterBottomLefts[i][0][1] - clusterTopLefts[k][0][1]) > distanceThresholds[2] - 5)):
    #                     print(distanceThresholds[2])
    #                     print(clusterArray[i])
    #                     print(clusterArray[k])
    #                     print("\n")
    #                     clusterDict[i] = "Category"
    #                     clusterDict[k] = "Menu Item"

    # #clusterDict[2] = "MenuDescription"
    # for k in kArr:
    #     clusterDict[k] = "Menu Description"
    # for i in range(len(clusterDict.values())):
    #     if(clusterDict[i] == None):
    #         clusterDict[i] = "Menu Item"

    for i in range(len(clusterDict.values())):
        #print(clusterArray[i])
        #print(clusterDict[i])
        #print("\n")
        n = 0
    #print(distanceThresholds)
    return clusterDict, priceClusterDict

def findThresholds(imgPath, clusterArray, clusterDict, wordVertices, numClusters):
    img = cv2.imread(imgPath)
    height = img.shape[0]
    width = img.shape[1]
    integerVertices = vertexArrCreator(wordVertices)
    #print(integerVertices)
    indexCounter = 1
    clusterInitIndex = []
    for i in range(len(clusterArray)):
        clusterInitIndex.append(indexCounter)
        for j in range(len(clusterArray[i])):
            indexCounter += 1
            #lineHeight = integerVertices[indexCounter][0][1] - integerVertices[indexCounter + len(clusterArray[i])][0][1]
        #for k in range(i+1, i+5):
            #k = 2


    # for i in range(len(clusterInitIndex)):
    #     print(clusterInitIndex[i])
    #     print()
    indexCounter = 1
    allWordAPC = []                             #cluster area per character
    
    for i in range(len(clusterArray)):
        wordClusterAPC = []
        charCounter = 0
        firstBottomLeft = integerVertices[indexCounter][0]
        lastTopRight = integerVertices[indexCounter + len(clusterArray[i]) - 1][-2]
        clusterArea = (abs(firstBottomLeft[0] - lastTopRight[0])) * (abs(firstBottomLeft[1] - lastTopRight[1]))
        for j in range(len(clusterArray[i])):
            wordBottomLeft = integerVertices[indexCounter][0]
            #print(wordBottomLeft)
            wordTopRight = integerVertices[indexCounter][-2]
            wordArea = (abs(wordTopRight[0] - wordBottomLeft[0])) * (abs(wordTopRight[1] - wordBottomLeft[1]))
            #print(wordArea)
            
            for char in clusterArray[i][j]:
                charCounter += 1
            indexCounter += 1
            valWordAPC = wordArea/(charCounter + len(clusterArray[i]) - 1)
            wordClusterAPC.append(valWordAPC)
        clusterAPC = round(clusterArea/charCounter,2)
        allWordAPC.append(clusterAPC)



    allClusterLineHeightArr = []
    for i in range(len(clusterInitIndex) - 1):
        minClusterLineHeightArr = []
        minClusterLineHeight = 10000
        for j in range(i + 1,len(clusterInitIndex)):
            #print(abs(integerVertices[clusterInitIndex[i]][-1][0] - integerVertices[clusterInitIndex[j]][-1][0]))
            if((clusterDict[i] != "Price" and clusterDict[j] != "Price" and (abs(integerVertices[clusterInitIndex[i]][-1][0] - integerVertices[clusterInitIndex[j]][-1][0])<0.4*width))): #change 200 to 1/2 of picture image length
                clusterLineHeight = abs(integerVertices[clusterInitIndex[i]][0][1] - integerVertices[clusterInitIndex[j]][-1][1])
                #print(clusterLineHeight)
                if(clusterLineHeight < minClusterLineHeight):
                    minClusterLineHeight = clusterLineHeight
                    #print([clusterArray[i], allWordAPC[i], clusterArray[j], allWordAPC[j]])
                    #print(minClusterLineHeight)
                    allClusterLineHeightArr.append(minClusterLineHeight)

            #clusterLineHeightArr.append(clusterLineHeight)
        #print(minClusterLineHeight)
        #allClusterLineHeightArr.append(minClusterLineHeight)


    X = np.empty([len(allClusterLineHeightArr),2])
    for i in range(len(allClusterLineHeightArr)):
        X[i] = [0,allClusterLineHeightArr[i]]
        #print(tempArr)
        #X = np.append(X,tempArr, axis = 0)

    # #Agglomerative Clustering
    # clustering = AgglomerativeClustering(n_clusters=2, linkage='single').fit(X)
    # clusterZero = []
    # clusterOne = []
    # clusterTwo = []
    # for i in range(len(clustering.labels_)):
    #     if(clustering.labels_[i] == 0):
    #         #print(X[i])
    #         clusterZero.append(X[i][1])
    #         #print("0")
    #     if(clustering.labels_[i] == 1):
    #         #print(X[i])
    #         #print("1")
    #         clusterOne.append(X[i][1])
    #     if(clustering.labels_[i] == 2):
    #         clusterTwo.append(X[i][1])
    #         #print(X[i])
    #         #print("2")
    # #print(clustering.labels_)
    # clusterZero = sorted(clusterZero)
    # clusterOne = sorted(clusterOne)
    # clusterTwo = sorted(clusterTwo)
    # print(clusterZero)
    # print(clusterOne)
    # print(clusterTwo)

    
    # dendrogram = hierarchy.dendrogram(hierarchy.linkage(X, method='single'))
    # plt.figure()
    # plt.show()


    #DBSCAN
    # neigh = NearestNeighbors(n_neighbors = 2)
    # nbrs = neigh.fit(X)
    # distances, indices = nbrs.kneighbors(X)
    # distances = np.sort(distances, axis = 0)
    # distances = distances[:,1]
    # plt.plot(distances)
    # plt.show()
    # m = DBSCAN(eps=30, min_samples=5)
    # m.fit(X)
    # clusters = m.labels_
    # colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid']
    # vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
    # plt.scatter(X[:,0], X[:,1], c=vectorizer(clusters))
    # plt.show()


    kmeans = KMeans(n_clusters=numClusters, random_state=0).fit(X)
    clusterCheck = kmeans.cluster_centers_
    #print(clusterCheck)

    kmeans = KMeans(n_clusters=numClusters, random_state=0).fit(X)
    clusters = kmeans.cluster_centers_
    #print(kmeans.cluster_centers_)

    clusters = [clusters[i][1] for i in range(len(clusters))]

    sortedCluster = sorted(clusters)
    #print(sortedCluster)
    #print(allClusterLineHeightArr)
    minClusterLineHeightDict = {}
    for num in allClusterLineHeightArr:
        if(num not in minClusterLineHeightDict):
            minClusterLineHeightDict[num] = 1
        else:
            minClusterLineHeightDict[num] = minClusterLineHeightDict[num] + 1
    #print(minClusterLineHeightDict)
    sortedDict = {k:v for k,v in sorted(minClusterLineHeightDict.items(), key = lambda item: item[1], reverse = True)}
    distanceArr = [list(sortedDict.keys())[0]]

    #print(sortedDict)
    #print(distanceArr)
    #print(sortedDict.keys())
    for key in sortedDict.keys():
        if(all((key > distance + 4 or key < distance - 4) for distance in distanceArr)):
            distanceArr.append(key)

            #distanceArr.append(key)
    #print(distanceArr)

    dArr = distanceArr[:3]
    sortedArr = sorted(dArr)
    #print(sortedArr)

    d1 = sortedArr[0] #distance between menu item and menu description
    d2 = sortedArr[1] #distance between menu description and next menu item
    d3 = sortedArr[2] #distance between category and menu item
    #print(dArr)

    return sortedCluster

def findWordAPC(clusterArray, wordVertices):
    integerVertices = vertexArrCreator(wordVertices)
    indexCounter = 1
    allWordAPC = []                             #cluster area per character
    
    for i in range(len(clusterArray)):
        wordClusterAPC = []
        #charCounter = 0
        #firstBottomLeft = integerVertices[indexCounter][0]
        #lastTopRight = integerVertices[indexCounter + len(clusterArray[i]) - 1][-2]
        for j in range(len(clusterArray[i])):
            charCounter = 0
            wordBottomLeft = integerVertices[indexCounter][0]
            #print(wordBottomLeft)
            wordTopRight = integerVertices[indexCounter][-2]
            wordArea = (abs(wordTopRight[0] - wordBottomLeft[0])) * (abs(wordTopRight[1] - wordBottomLeft[1]))
            #print(wordArea)
            for char in clusterArray[i][j]:
                charCounter += 1
            indexCounter += 1
            valWordAPC = round(wordArea/charCounter,2)
            wordClusterAPC.append(valWordAPC)
        allWordAPC.append(wordClusterAPC)
    medianClusterAPC = [statistics.median(cluster) for cluster in allWordAPC]
    #print(medianClusterAPC)
    for i in range(len(medianClusterAPC)):
        n = 1
        #print(clusterArray[i], medianClusterAPC[i])
    sortedMedianClusterAPC = sorted(medianClusterAPC)
    initValue = sortedMedianClusterAPC[0]
    overallFontGroup = []
    fontGroup = []
    for i in range(len(sortedMedianClusterAPC)):
        if(sortedMedianClusterAPC[i] < 1.3 * initValue):
            fontGroup.append(sortedMedianClusterAPC[i])
            #print(sortedMedianClusterAPC[i])
        else:
            initValue = sortedMedianClusterAPC[i]
            #print(initValue)
            overallFontGroup.append(fontGroup)
            fontGroup = [initValue]
    overallFontGroup.append(fontGroup)

    X = np.empty([len(sortedMedianClusterAPC),2])
    for i in range(len(sortedMedianClusterAPC)):
        X[i] = [0,sortedMedianClusterAPC[i]]
        #print(tempArr)
        #X = np.append(X,tempArr, axis = 0)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    clusters = kmeans.cluster_centers_
    #print(kmeans.cluster_centers_)

    cluster1 = clusters[0][1]
    cluster2 = clusters[1][1]
    cluster3 = clusters[2][1]
    clusters = [cluster1,cluster2,cluster3]

    sortedCluster = sorted(clusters)
    #print(sortedCluster)
    #print(sortedMedianClusterAPC)
    #print(overallFontGroup)
    #print(medianClusterAPC)
    # for i in range(len(medianClusterAPC)):
    #    print(clusterArray[i])
    #    print(medianClusterAPC[i])
    #print(sortedMedianClusterAPC)
    #print(overallFontGroup)
    return overallFontGroup,medianClusterAPC
    #print(allWordAPC)
    #print(clusterArray)

        

    #print(sortedDict)

    
    #print(allClusterLineHeightArr)
    # print(len(clusterDict))
    # print(sum(clusterDict[i]!="Price" for i in range(len(clusterDict))))
    # print(len(allClusterLineHeightArr))
    #print(clusterArray)


            
            


def normalizeDimensions(wordVertices):
    integerVertices = vertexArrCreator(wordVertices)
    maxXDim = 0
    maxYDim = 0
    # for i in range(len(integerVertices)):

#priceClassification(imageText)


def drawWordBoxes(imgPath, wordText, wordVertices):
    img = cv2.imread(imgPath)
    integerVertices = vertexArrCreator(wordVertices)
    for i in range(len(wordText)):
        wordBottomLeft = integerVertices[i][0]
        wordTopRight = integerVertices[i][-2]
        cv2.rectangle(img, wordBottomLeft, wordTopRight, color = (0,0,0))
    cv2.imshow('img',img)
    cv2.waitKey(0)
    output_path = "/Users/sohambose/Harvard/Sleek/MenuDataset/simpleImages"
    file_name = os.path.basename(imgPath).split('.')[0]
    file_name = file_name.split()[0]
    save_path = os.path.join(output_path, file_name + "_filter_" + ".jpg")
    cv2.imwrite(save_path, img)


def drawBoxes(imgPath, clusterArray, clusterDict, priceClusterDict, wordText, wordVertices, associationDict, bottomLeftCoordinates, topRightCoordinates, associationFlag):
    img = cv2.imread(imgPath)
    integerVertices = vertexArrCreator(wordVertices)
    rotatedCoordinates = boxCoordinateRotation(integerVertices)
    #print(clusterArray)
    #print(clusterDict)



    indexCounter = 1
    for i in range(len(priceClusterDict)):
        firstBottomLeft = integerVertices[indexCounter][0]
        firstTopLeft = integerVertices[indexCounter][-1]
        lastTopRight = integerVertices[indexCounter + len(clusterArray[i]) - 1][-2]
        lastBottomRight = integerVertices[indexCounter + len(clusterArray[i]) - 1][1]
        if(priceClusterDict[i] == "Price"):
            boxColor = (255,0,0)
            cv2.line(img, firstBottomLeft, lastBottomRight, color = boxColor)
            cv2.line(img, lastBottomRight, lastTopRight, color = boxColor)
            cv2.line(img, lastTopRight, firstTopLeft, color = boxColor)
            cv2.line(img, firstTopLeft, firstBottomLeft, color = boxColor)

        for j in range(len(clusterArray[i])):
            indexCounter += 1

    indexCounter = 1
    for i in range(len(clusterArray)):
        firstBottomLeft = integerVertices[indexCounter][0]
        firstTopLeft = integerVertices[indexCounter][-1]
        lastTopRight = integerVertices[indexCounter + len(clusterArray[i]) - 1][-2]
        lastBottomRight = integerVertices[indexCounter + len(clusterArray[i]) - 1][1]

        if(clusterDict[i] == "Price"): #blue
            boxColor = (255,0,0)
        elif(clusterDict[i] == "Category"): #green
            boxColor = (0,255,0)
        elif(clusterDict[i] == "Menu Description"): #red
            boxColor = (0,0,255)
        elif(clusterDict[i] == "Menu Item"):#purple
            boxColor = (128,0,128)
        else:
            boxColor = (0,0,0)
        cv2.line(img, firstBottomLeft, lastBottomRight, color = boxColor)
        cv2.line(img, lastBottomRight, lastTopRight, color = boxColor)
        cv2.line(img, lastTopRight, firstTopLeft, color = boxColor)
        cv2.line(img, firstTopLeft, firstBottomLeft, color = boxColor)
        #print(boxColor)
        # if(clusterDict[i] == None):
        #     print(clusterArray[i])

        for j in range(len(clusterArray[i])):
            indexCounter += 1

    


    #draw association boxes
    if(associationFlag):
        for i in range(len(topRightCoordinates)):
            cv2.rectangle(img, bottomLeftCoordinates[i], topRightCoordinates[i], color = (0,0,0))


    
    mapperDict = {}
    indexCounter = 1
    for i in range(len(clusterArray)):
        mapperDict[i] = indexCounter
        indexCounter += len(clusterArray[i])
    #print(mapperDict)

    associationIndex = []
    for key in associationDict:
        indexArr = []
        for i in range(len(associationDict[key])):
            clusterIndex = clusterArray.index(associationDict[key][i])
            indexArr.append(clusterIndex)
        associationIndex.append(indexArr)
    #print(clusterArray.index(['3.50']))
    #print(associationIndex)

    #print(mapperDict)
    #print(associationIndex)
    blockedIndex = []
    for block in associationIndex:
        mappedIndex = []
        for index in block:
            #print(index)
            #print(index + len(clusterArray[index]) - 1)
            if(index<= len(mapperDict) and (index + len(clusterArray[index]) - 1 <= len(mapperDict))):
                mappedIndex.append([integerVertices[mapperDict[index]], integerVertices[mapperDict[index + len(clusterArray[index]) - 1]]]) #vertices of the first and last boxes of the cluster
            #cv2.rectangle(img, integerVertices[mapperDict[index]][0], integerVertices[mapperDict[index]][-2], color  =(0,0,0))
            #print(clusterArray[40])
        #print(mappedIndex)
        blockedIndex.append(mappedIndex)

    blockCoord = []
    for block in blockedIndex:
        clusterCoord = []
        for cluster in block:
            #print(cluster)
            rectCoord = []
            for i in range(len(cluster)):
                rectCoord.append(cluster[i][0]) #bottom left corner of each box in the cluster(first and last)
                rectCoord.append(cluster[i][2]) #top right corner of each box in the cluster(first and last)
                #cv2.rectangle(img, cluster[i][0], cluster[i][2], color = (0,0,0))
                #print(rectCoord)
            #print(rectCoord)
            clusterCoord.append(rectCoord)
        blockCoord.append(clusterCoord)

    


    #print(blockCoord)
    
    # for block in blockCoord:
    #     for coords in block:
    #         bottomLeftCoord = block[0][0]
    #         topRightCoord = block[1][len(block[1]) - 1]
    #         cv2.rectangle(img, bottomLeftCoord, topRightCoord, color = (0,0,0))
        




                



    #print(blockedIndex)

        #print(indexArr)
    # indexCounter = 1
    # for i in range(len(clusterArray)):
    #     firstBottomLeft = rotatedCoordinates[indexCounter][0]
    #     firstTopLeft = rotatedCoordinates[indexCounter][-1]
    #     lastTopRight = rotatedCoordinates[indexCounter + len(clusterArray[i]) - 1][-2]
    #     lastBottomRight = rotatedCoordinates[indexCounter + len(clusterArray[i]) - 1][1]
    #     if(clusterDict[i] == "Price"): #blue
    #         cv2.line(img, firstBottomLeft, lastBottomRight, color = (255,0,0))
    #         cv2.line(img, lastBottomRight, lastTopRight, color = (255,0,0))
    #         cv2.line(img, lastTopRight, firstTopLeft, color = (255,0,0))
    #         cv2.line(img, firstTopLeft, firstBottomLeft, color = (255,0,0))
    #     if(clusterDict[i] == "Category"): #green
    #         cv2.line(img, firstBottomLeft, lastBottomRight, color = (0,255,0))
    #         cv2.line(img, lastBottomRight, lastTopRight, color = (0,255,0))
    #         cv2.line(img, lastTopRight, firstTopLeft, color = (0,255,0))
    #         cv2.line(img, firstTopLeft, firstBottomLeft, color = (0,255,0))
    #     if(clusterDict[i] == "Menu Description"): #red
    #         cv2.line(img, firstBottomLeft, lastBottomRight, color = (0,0,255))
    #         cv2.line(img, lastBottomRight, lastTopRight, color = (0,0,255))
    #         cv2.line(img, lastTopRight, firstTopLeft, color = (0,0,255))
    #         cv2.line(img, firstTopLeft, firstBottomLeft, color = (0,0,255))
    #     if(clusterDict[i] == "Menu Item"):#purple
    #         cv2.line(img, firstBottomLeft, lastBottomRight, color = (128,0,128))
    #         cv2.line(img, lastBottomRight, lastTopRight, color = (128,0,128))
    #         cv2.line(img, lastTopRight, firstTopLeft, color = (128,0,128))
    #         cv2.line(img, firstTopLeft, firstBottomLeft, color = (128,0,128))

    #     for j in range(len(clusterArray[i])):
    #         indexCounter += 1
    
    #print(clusterDict.values())
    
    cv2.imshow('img',img)
    cv2.waitKey(0)
    output_path = "/Users/sohambose/Harvard/Sleek/MenuDataset/simpleImages"
    file_name = os.path.basename(imgPath).split('.')[0]
    file_name = file_name.split()[0]
    save_path = os.path.join(output_path, file_name + "_filter_" + ".jpg")
    cv2.imwrite(save_path, img)
    print(save_path)

#def agglomerativeClustering(data):




def associationClusters(clusterDict, clusterArray, integerVertices):
    associationDict = {}
    temp = list(clusterDict)
    mapperDict = {}
    indexCounter = 1
    for i in range(len(clusterArray)):
        mapperDict[i] = indexCounter
        indexCounter += len(clusterArray[i])
    #print(mapperDict)
    temp2 = list(mapperDict)
    #print(temp, temp2)
    clusterCoordinates = []
    for key in temp2:
        indexCounter = mapperDict[key]
        singleClusterCoordinates = []
        firstBottomLeft = integerVertices[indexCounter][0]
        firstTopLeft = integerVertices[indexCounter][-1]
        lastTopRight = integerVertices[indexCounter + len(clusterArray[i]) - 1][-2]
        lastBottomRight = integerVertices[indexCounter + len(clusterArray[i]) - 1][1]
        singleClusterCoordinates.append(firstBottomLeft)
        singleClusterCoordinates.append(firstTopLeft)
        singleClusterCoordinates.append(lastTopRight)
        singleClusterCoordinates.append(lastBottomRight)
        clusterCoordinates.append(singleClusterCoordinates)

    associationCoordinates = []
    
    for key in temp:
        singleAssociationCoordinates = []
        result = []
        if(clusterDict[key] == "Category"):
            categoryArr = clusterArray[key]
        if(clusterDict[key] == "Menu Item"):
            singleAssociationCoordinates = []
            singleAssociationCoordinates.append(clusterCoordinates[key])
            #print(clusterArray[key])
            result = [clusterArray[key]]
            try:
                nextKey = temp[temp.index(key) + 1]
            except(ValueError, IndexError):
                nextKey = None
            while(nextKey != None and clusterDict[nextKey] != "Menu Item" and clusterDict[nextKey] != None and clusterDict[nextKey] != "Category"):
                result.append(clusterArray[nextKey])
                singleAssociationCoordinates.append(clusterCoordinates[nextKey])
                try: 
                    nextKey = temp[temp.index(nextKey) + 1]
                except(ValueError,IndexError):
                    nextKey = None
            #result.append(categoryArr)
        if(len(singleAssociationCoordinates) != 0 and singleAssociationCoordinates != None):
            associationCoordinates.append(singleAssociationCoordinates)
        if(len(result) != 0):
            dictKey = ""
            for word in clusterArray[key]:
                dictKey += word
            associationDict[dictKey] = result
            #print(result)
    #print(associationDict)
    #print(associationCoordinates)
    #print("\n")
    #print("\n")
    return associationDict, associationCoordinates

def findAssociationBoundary(associationCoordinates):
    associationBoundaryCoordinates = []
    topRightCoordinates = []
    bottomLeftCoordinates = []
    for association in associationCoordinates:
        topLeft = association[0][0]
        bottomLeft = association[0][1]
        bottomRight = association[0][2]
        topRight = association[0][3]
        for clusterCoordinates in association:
            if(clusterCoordinates[0][1] > topLeft[1]):
                topLeft = clusterCoordinates[0]

            if(clusterCoordinates[1][0] <= bottomLeft[0]):
                bottomLeft = clusterCoordinates[1]

            if(clusterCoordinates[2][0] > bottomRight[0]):
                bottomRight = clusterCoordinates[2]
                #print(topRight)

            if(clusterCoordinates[3][0] > topRight[0]):
                topRight = clusterCoordinates[3]
        topRightCoordinates.append(topRight)
        bottomLeftCoordinates.append(bottomLeft)
    #print(associationBoundaryCoordinates)
    #print(topRightCoordinates)
    #print(bottomLeftCoordinates)
    return bottomLeftCoordinates, topRightCoordinates


    


def categoryAssociation(clusterDict, clusterArray,integerVertices):
    mapperDict = {}
    indexCounter = 1
    for i in range(len(clusterArray)):
        mapperDict[i] = indexCounter
        indexCounter += len(clusterArray[i])
    #print(mapperDict)
    categoryDict = {}
    categoryArray = []
    for key in clusterDict:
        if(clusterDict[key] == "Category"):
            categoryArray.append([mapperDict[key],clusterArray[key]])
    #print(categoryArray)
    for key in clusterDict:
        for index in range(len(categoryArray)-1):
            currCategoryCluster = integerVertices[categoryArray[index][0]]
            nextCategoryCluster = integerVertices[categoryArray[index + 1][0]]
            currClusterTopLeft = integerVertices[mapperDict[key]][-1]
            #print(currClusterTopLeft)
            #print(clusterArray[key])
            if(clusterDict[key] == "Menu Item" and currClusterTopLeft[1]>currCategoryCluster[-1][1] and currClusterTopLeft[0] < nextCategoryCluster[-1][0]):
                print(clusterArray[key])
            #if(currCategoryCluster[-1][1] < )
            if(clusterDict[key] == "Menu Item" and currClusterTopLeft[1] >  currCategoryCluster[-1][1] and currClusterTopLeft[1] < nextCategoryCluster[-1][1] and currClusterTopLeft[0] < nextCategoryCluster[-1][0]):
                currAssociation = clusterArray[key]
                mappedIndex = categoryArray[index][0]
                currAssociation.append(clusterArray[list(mapperDict.keys())[list(mapperDict.values()).index(mappedIndex)]])
                print(currAssociation)
    print(categoryDict)
    #print(categoryArray)



                

imgPath = '/Users/sohambose/Harvard/Sleek/MenuDataset/simpleImages/Simple-Italian-Restaurant-Menu-Template.jpeg'
def main(imgPath):
    imageText,imageVertices = detect_text(imgPath)
    integerVertices = vertexArrCreator(imageVertices)
    angle =  boxCoordinateRotation(integerVertices)
    if(angle != 0):
        print(angle)
        rotatedImage,imgPath = uncroppedRotation(imgPath, angle)
        imageText, imageVertices = detect_text(imgPath)
        integerVertices = vertexArrCreator(imageVertices)
    clusterArray = geometricClustering(imageText, imageVertices) 
    #print(clusterArray)
    clusterDict = priceClassification(clusterArray, imageText, imageVertices)
    distanceThresholds = findThresholds(imgPath, clusterArray, clusterDict, imageVertices, 3)
    #print(distanceThresholds)
    overallFontGroup = findWordAPC(clusterArray, imageVertices)[0]
    clusterDict, priceClusterDict = overallClassification(imgPath, clusterArray, imageText, imageVertices, clusterDict, distanceThresholds, overallFontGroup)
    print(clusterDict)
    associationDict, associationCoordinates = associationClusters(clusterDict, clusterArray, integerVertices)
    #print(associationDict)
    bottomLeftCoordinates, topRightCoordinates = findAssociationBoundary(associationCoordinates)
    #vertexArray = vertexArrCreator(imageVertices)
    #rotatedCoordinates = boxCoordinateRotation(vertexArray)
    drawBoxes(imgPath, clusterArray, clusterDict, priceClusterDict, imageText, imageVertices, associationDict, bottomLeftCoordinates, topRightCoordinates, False)
    #drawWordBoxes(imgPath, imageText, imageVertices)
    #categoryAssociation(clusterDict,clusterArray, integerVertices)
main(imgPath)




