import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pyocr
import pyocr.builders
import codecs
from PIL import Image
import tesserocr
from tesserocr import PyTessBaseAPI
import fnmatch
import easyocr

result = {}

# print(len(pyocr.get_available_tools()))
# tool = pyocr.get_available_tools()[0]
# print(tool)
# builder = pyocr.builders.TextBuilder()
# langs = tool.get_available_languages()
# lang = langs[0]

# txt = tool.image_to_string(
#     Image.open('/Users/sohambose/Harvard/Sleek/MenuDataset/scannedImages/385NorthMenuScanned.jpeg'),
#     lang=lang,
#     builder=builder)

# with codecs.open("toto.txt", 'w', encoding='utf-8') as file_descriptor:
#     builder.write_file(file_descriptor, txt)



#saves grayscale image, finds text in image, outputs boxed version of image
def get_string(imgPath, outputDir):
    # Read image using opencv
    img = cv2.imread(imgPath)
    img2 = cv2.imread(imgPath)
    file_name = os.path.basename(imgPath).split('.')[0]
    file_name = file_name.split()[0]

    output_path = os.path.join(outputDir, file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Crop the areas where provision number is more likely present
    #img = crop_image(img, pnr_area[0], pnr_area[1], pnr_area[2], pnr_area[3])
    # img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) #imgf contains Binary image

    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    save_path = os.path.join(output_path, file_name + "_filter_" + ".jpg")
    cv2.imwrite(save_path, img)

    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(img, lang="eng")

    save_path_results = os.path.join(output_path, file_name + "_filter_" + ".txt")
    f = open(save_path_results, 'w')
    f.write(result)
    f.close()

    #  Apply threshold to get image with only black and white
    #img = apply_threshold(img, method)
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    numBoxes = len(d['level'])
    for i in range(numBoxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        if(int(d['conf'][i]) == -1):
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            cv2.rectangle(img2, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 2)


    save_path = os.path.join(output_path, file_name + "_boxed_" + ".jpg")
    cv2.imwrite(save_path, img2)


    return result

#get_string('/Users/sohambose/Harvard/Sleek/MenuDataset/preprocessedImages/boxedImages/385NorthMenuScanned/385NorthMenuScanned8_boxed_.jpg','/Users/sohambose/Harvard/Sleek/MenuDataset/boxedImages/scannedImages')

preprocessedDir = '/Users/sohambose/Harvard/Sleek/MenuDataset/preprocessedImages'
overallDir = '/Users/sohambose/Harvard/Sleek/MenuDataset'


#runs all images in a directory through get_string
def preprocessing(overallDirectory, outputDirectory):
    for folder in os.listdir(overallDirectory):
        if(folder != '.DS_Store'):
            pictureDir = os.path.join(overallDirectory,folder)
            outputDir = os.path.join(outputDirectory, folder)
        for filename in os.listdir(pictureDir):
            if(filename.endswith(".jpeg") or filename.endswith(".png")):
                f = os.path.join(pictureDir, filename)
                get_string(f,outputDir)

#preprocessing(overallDir, preprocessedDir)

def cropBoxes(imgPath, outputDir):
    img = cv2.imread(imgPath)
    file_name = os.path.basename(imgPath).split('.')[0]
    file_name = file_name.split()[0]

    output_path = os.path.join(outputDir, file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    row, col = img.shape[:2]
    bottom = img[row-2:row, 0:col]
    mean = cv2.mean(bottom)[0]

    bordersize = 10
    border = cv2.copyMakeBorder(
        img,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[mean, mean, mean]
    )

    #cv2.imshow('border', border)
    #cv2.waitKey(0)

    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    numBoxes = len(d['level'])
    #print(d['conf'])
    #print(len(d['conf']))
    for i in range(numBoxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        if(int(d['conf'][i]) != -1):
            box = border[y + 8: y + h + 12, x + 8: x + w + 12]
            #cv2.rectangle(img2, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 2)
            save_path = os.path.join(output_path, file_name + str(i) + "_boxed_" + ".jpg")
            #print(numBoxes)
            cv2.imwrite(save_path, box)
            #print(i)

#cropBoxes('/Users/sohambose/Harvard/Sleek/MenuDataset/scannedImages/385NorthMenuScanned.jpeg','/Users/sohambose/Harvard/Sleek/MenuDataset/preprocessedImages/boxedImages')

def cropPreprocessing(overallDirectory, outputDirectory):
    for folder in os.listdir(overallDirectory):
        if(folder != '.DS_Store'):
            pictureDir = os.path.join(overallDirectory,folder)
            outputDir = os.path.join(outputDirectory, folder)
            #if not os.path.exists(outputDir):
                #os.makedirs(outputDir)
        for filename in os.listdir(pictureDir):
            if(filename.endswith(".jpeg") or filename.endswith(".png")):
                #croppedDir = os.path.join(pictureDir, filename)
                f = os.path.join(pictureDir, filename)
                cropBoxes(f,outputDir)
#cropPreprocessing(overallDir, '/Users/sohambose/Harvard/Sleek/MenuDataset/boxedImages')

def createCroppedFile(overallDirectory, outputDirectory):
    for folder in os.listdir(overallDirectory):
        if(folder != '.DS_Store'):
            pictureDir = os.path.join(overallDirectory,folder)
            outputDir = os.path.join(outputDirectory, folder)
            for embeddedFolder in os.listdir(pictureDir):
                if(embeddedFolder != '.DS_Store'):
                    embeddedPictureDir = os.path.join(pictureDir,embeddedFolder)
                    embeddedOutputDir = os.path.join(outputDir,embeddedFolder)
                    save_path_results = os.path.join(outputDir, embeddedFolder + "_boxText_" + ".txt")
                    txtFile = open(save_path_results, 'w')
                    for filename in os.listdir(embeddedPictureDir):
                        if(filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg")):
                            f = os.path.join(embeddedPictureDir,filename)
                            img = cv2.imread(f)
                            boxText = pytesseract.image_to_string(img, lang="eng")
                            txtFile.write(boxText)
                    txtFile.close()


        #     if not os.path.exists(outputDir):
        #         os.makedirs(outputDir)
        # for filename in os.listdir(pictureDir):
        #     if(filename.endswith(".jpeg") or filename.endswith(".png")):
        #         #croppedDir = os.path.join(pictureDir, filename)
        #         f = os.path.join(pictureDir, filename)
        #         cropBoxes(f,outputDir)
#createCroppedFile('/Users/sohambose/Harvard/Sleek/MenuDataset/boxedImages','/Users/sohambose/Harvard/Sleek/MenuDataset/boxedImages')
#cropPreprocessing(overallDir, '/Users/sohambose/Harvard/Sleek/MenuDataset/preprocessedImages/boxedImages')


def randomMethod(directory):
    save_path_results = os.path.join(directory + "/385NorthMenuScanned_textBox" + ".txt")
    print(save_path_results)
    txtFile = open(save_path_results, 'w')
    for filename in os.listdir(directory):
        if(filename.endswith(".jpg")):
            imgPath = os.path.join(directory, filename)
            #print(type(imgPath))
            print(imgPath)
            img = cv2.imread(imgPath)
            imgString = pytesseract.image_to_string(img, lang = "eng")
            #print(type(imgString))
            #txtFile.write("a")
            txtFile.write(imgString)
            txtFile.write("\n")
            print(imgString)
    txtFile.close()
    return 0

#randomMethod('/Users/sohambose/Harvard/Sleek/MenuDataset/boxedImages/scannedImages/385NorthMenuScanned')


# pictureDir = '/Users/sohambose/Harvard/Sleek/MenuDataset/scannedImages'
# for filename in os.listdir(pictureDir):
#     if(filename.endswith(".jpeg") or filename.endswith(".png")):
#         f = os.path.join(pictureDir, filename)
#         get_string(f,1)
#get_string('/Users/sohambose/Harvard/Sleek/MenuDataset/scannedImages/theStandMenuPicture.jpeg', 1)

#img_path = '/Users/sohambose/Harvard/Sleek/MenuDataset/preprocessedImages/Scanned Images/theStandMenuPicture/theStandMenuPicture_filter_.jpg'
#img = cv2.imread(img_path)
#d = pytesseract.image_to_data(img, output_type=Output.DICT)
#print(d['conf'])
# n_boxes = len(d['text'])
# for i in range(n_boxes):
#     if int(d['conf'][i]) > 0:
#         (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#         img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# n_boxes = len(d['level'])
# for i in range(n_boxes):
#     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# cv2.imshow('img', img)
# cv2.waitKey(0)



# image = cv2.imread('/Users/sohambose/Harvard/Sleek/MenuDataset/scannedImages/385NorthMenuScanned.jpeg')

# # get grayscale image
# def get_grayscale(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# img = '/Users/sohambose/Harvard/Sleek/MenuDataset/scannedImages/385NorthMenuScanned.jpeg'
# with PyTessBaseAPI() as api:
#         api.SetImageFile(img)
#         print(api.GetUTF8Text())
#         print(api.AllWordConfidences())



#using pyocr
#print(len(pyocr.get_available_tools()))
# tool = pyocr.get_available_tools()[0]
# print(tool)
# builder = pyocr.builders.TextBuilder()
# langs = tool.get_available_languages()
# lang = langs[0]

# txt = tool.image_to_string(
#     Image.open('/Users/sohambose/Harvard/Sleek/MenuDataset/scannedImages/385NorthMenuScanned.jpeg'),
#     lang=lang,
#     builder=builder)

# with codecs.open("/Users/sohambose/Harvard/Sleek/MenuDataset" + "/doodoo.txt", 'w', encoding='utf-8') as file_descriptor:
#     builder.write_file(file_descriptor, txt)

def pyocrImageToString(imgPath, outputDir, toolNum):
    tool = pyocr.get_available_tools()[toolNum]
    builder = pyocr.builders.TextBuilder()
    langs = tool.get_available_languages()
    lang = langs[0]

    txt = tool.image_to_string(
    Image.open(imgPath),
    lang=lang,
    builder=builder)

    file_name = os.path.basename(imgPath).split('.')[0]
    file_name = file_name.split()[0]

    output_path = os.path.join(outputDir, file_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    txtPath = os.path.join(output_path, file_name + ".txt")

    with codecs.open(txtPath, 'w', encoding='utf-8') as file_descriptor:
        builder.write_file(file_descriptor, txt)

def pyocrIterator(overallDirectory, outputDirectory, toolNum):
    for folder in os.listdir(overallDirectory):
        if(folder != '.DS_Store'):
            pictureDir = os.path.join(overallDirectory,folder)
            outputDir = os.path.join(outputDirectory, folder)
        for filename in os.listdir(pictureDir):
            if(filename.endswith(".jpeg") or filename.endswith(".png")):
                f = os.path.join(pictureDir, filename)
                pyocrImageToString(f,outputDir, toolNum)

#pyocrImageToString('/Users/sohambose/Harvard/Sleek/MenuDataset/scannedImages/385NorthMenuScanned.jpeg','/Users/sohambose/Harvard/Sleek/MenuDataset/pyOCR',0)
#pyocrIterator('/Users/sohambose/Harvard/Sleek/MenuDataset', '/Users/sohambose/Harvard/Sleek/MenuDataset/pyOCR',0)



#easyocr
reader = easyocr.Reader(['en'], gpu = False) # need to run only once to load model into memory
result = reader.readtext('/Users/sohambose/Harvard/Sleek/MenuDataset/preprocessedImages/scannedImages/385NorthMenuScanned/385NorthMenuScanned_filter_.jpg')
print(result)

