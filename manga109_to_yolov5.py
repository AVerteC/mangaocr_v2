import concurrent.futures
import pprint
import manga109api
import math
import shutil
import time
import numpy as np
import os
import random

np.set_printoptions(linewidth=1000)

manga109_root_dir = "Manga109_released_2021_02_28"
p = manga109api.Parser(root_dir=manga109_root_dir)


# notes
# nc: 4  # number of classes
# names: ['body', 'frame', 'face', 'text']  # class names
# (if no objects in image, no *.txt file is required)

def readPageLabel(page, labelType, labelValue):
    # labelValues "body": 0, "frame": 1, "face": 2, "text": 3
    labelDict = page.get(labelType)
    # un-normalized data
    labelArray = []
    if len(labelDict) == 0:
        #         print("nothing found for this page and label", labelNames.get(labelType), "combination")
        return None
    for label in labelDict:
        #       un normalized h/w
        unHeight = page.get('@height')
        unWidth = page.get('@width')
        xmax = label.get('@xmax')
        xmin = label.get('@xmin')
        ymax = label.get('@ymax')
        ymin = label.get('@ymin')
        className = labelValue
        xcenter = ((xmax + xmin) / 2) / unWidth
        ycenter = ((ymax + ymin) / 2) / unHeight
        width = math.sqrt(xmax * xmax - xmin * xmin) / unWidth
        height = math.sqrt(ymax * ymax - ymin * ymin) / unHeight
        yoloText = str(className) + " " + str(xcenter) + " " + str(ycenter) + " " + str(width) + " " + str(height)
        labelArray.append(yoloText)
    return labelArray


def readBook(bookName, body, frame, face, text):
    bookAnnotation = p.get_annotation(book=bookName)
    # use dict because numpy matrix row has to have same length
    bookPageLabels = {}
    pageNumber = 0
    for page in bookAnnotation["page"]:
        pageLabels = []
        if body:
            pageLabels.append(readPageLabel(page, "body", 9))
        if frame:
            pageLabels.append(readPageLabel(page, "frame", 0))
        if face:
            face = readPageLabel(page, "face", 9)
            pageLabels.append(face)
        if text:
            text = readPageLabel(page, "text", 1)
            pageLabels.append(text)
        bookPageLabels[pageNumber] = pageLabels
        pageNumber += 1
    return bookPageLabels


def pageCount(bookName):
    bookAnnotation = p.get_annotation(book=bookName)
    return len(bookAnnotation["page"])


def splitNumber(trainP, validateP, testP, totalItems):
    if (trainP + validateP + testP) > 100:
        raise Exception("train% and validate% sum > 100")
    if (trainP + validateP + testP) != 100:
        raise Exception("train% and validate% sum != 100")
    trainCount = math.floor(totalItems * (trainP / 100))
    validateCount = math.floor(totalItems * (validateP / 100))
    testCount = math.floor(totalItems * (testP / 100))
    return trainCount, validateCount, testCount


def imageCount(data):
    globalImgNumber = 0
    allBookLabels = []
    allImages = []
    for bookNumber in range(0, len(data)):
        # book = 0->108
        book = data[bookNumber]
        bookName = p.books[bookNumber]
        bookPageCount = pageCount(bookName)
        for pageNumber in range(0, bookPageCount):
            globalImgNumber += 1
    return (globalImgNumber)


# def outputFormattedData(data, ):

def formLabelList(data):
    globalImgNumber = 0
    allBookLabels = []
    allImages = []
    for bookNumber in range(0, len(data)):
        # book = 0->108
        book = data[bookNumber]
        bookName = p.books[bookNumber]
        bookPageCount = pageCount(bookName)
        #         print(bookName, str(bookPageCount))
        for pageNumber in range(0, bookPageCount):
            imgPath = p.img_path(book=bookName, index=pageNumber)
            #             print(imgPath)
            tgtPath = "datasets/Manga109/images/ImgTotal/" + "im" + str(globalImgNumber) + ".jpg"
            os.makedirs(os.path.dirname(tgtPath), exist_ok=True)
            shutil.copy(imgPath, tgtPath)
            #             shutil.copyfile(imgPath, "C:Users/Alan/Documents/DEV/MangaOCR/datasets/Manga109/ImgTotal/" + "im" + str(globalImgNumber) + ".jpg")
            page = book.get(pageNumber)
            #                 print("no labels found")
            if page != [None, None, None, None]:
                directory = "datasets/Manga109/labels/ImgLabels/"
                filename = "im" + str(globalImgNumber)
                labelTxt = open(directory + filename + ".txt", "w+")
                for label in page:
                    #                     print(label)
                    if label != None:
                        line = label[0] + "\n"
                        #                         print(line)
                        labelTxt.write(line)
                labelTxt.close()
        #                 print("next")
        #             print(" ")
        globalImgNumber += 1
    return


def outputPaths(data):
    globalImgNumber = 0
    allBookLabels = []
    allImages = []
    for bookNumber in range(0, len(data)):
        # book = 0->108
        book = data[bookNumber]
        bookName = p.books[bookNumber]
        bookPageCount = pageCount(bookName)
        for pageNumber in range(0, bookPageCount):
            imgPath = p.img_path(book=bookName, index=pageNumber)
            tgtFilename = "im" + str(globalImgNumber)
            allImages.append([imgPath, tgtFilename + ".jpg"])
            page = book.get(pageNumber)
            pageLabels = []
            if page == [None, None, None, None] or page == [None, None]:
                #                 print("page has no labels")
                pageLabels.append(None)
            else:
                #                 print("page:", len(page))
                for labels in page:
                    if labels != None:
                        for label in labels:
                            #                             print("labels:", label)
                            pageLabels.append(label + "\n")
            allBookLabels.append([tgtFilename + ".txt", pageLabels])
            globalImgNumber += 1
    return allImages, allBookLabels

def skippedPages(data):
    globalImgNumber = 0
    allBookLabels = []
    allImages = []
    skipped_pages = []
    for bookNumber in range(0, len(data)):
        # book = 0->108
        book = data[bookNumber]
        bookName = p.books[bookNumber]
        bookPageCount = pageCount(bookName)
        for pageNumber in range(0, bookPageCount):
            imgPath = p.img_path(book=bookName, index=pageNumber)
            tgtFilename = "im" + str(globalImgNumber)
            allImages.append([imgPath, tgtFilename + ".jpg"])
            page = book.get(pageNumber)
            pageLabels = []
            if page == [None, None, None, None] or page == [None, None]:
                filename = "im" + str(globalImgNumber) + ".jpg"
                skipped_pages.append(filename)
            # elif page[1]==None and page[3]==None:
            #     filename = "im" + str(globalImgNumber) + ".jpg"
            #     skipped_pages.append(filename)
            globalImgNumber += 1
    return skipped_pages




def readAllBooks():
    #     multithreading doesn't work for this
    allBookLabels = []
    for book in p.books:
        allBookLabels.append(readBook(book, False, True, False, True))
    return allBookLabels


def copyFile(source, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy(source, target)


def saveTxt(labelArray, target):
    print("saving file:", target)
    if len(labelArray) == 0:
        print("no feature to save for this image")
        return "image has no features to save"
    elif labelArray[0] is None:
        print("no feature to save for this image")
        return "image has no features to save"
    labelTxt = open(target, "w+")
    for label in labelArray:
        line = label
        labelTxt.write(line)
    labelTxt.close()
    return target


def cleanImgFilename(imgFilename):
    cleanImgFilename = imgFilename.replace("\\", "/")
    return cleanImgFilename


def threadSaveImg(i, targetFilename):
    sourceImg = cleanImgFilename(i[0])
    imgFilename = i[1]
    targetImg = targetFilename + imgFilename
    copyFile(sourceImg, targetImg)
    return targetImg


def threadSaveTxt(i, targetFilename):
    txtFilename = i[0]
    # print("saving file:", targetFilename)
    if txtFilename != None:
        txtTarget = targetFilename + txtFilename
        saveTxt(i[1], txtTarget)
    return txtTarget


def saveFormattedImg(dataArray, targetFilename):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in dataArray:
            futures.append(executor.submit(threadSaveImg, i=i, targetFilename=targetFilename))


#         for future in concurrent.futures.as_completed(futures):
#             print("saved " + future.result())

def saveFormattedTxt(dataArray, targetFilename):
    # for i in dataArray:
    #     threadSaveTxt(i, targetFilename)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in dataArray:
            futures.append(executor.submit(threadSaveTxt, i=i, targetFilename=targetFilename))


#         for future in concurrent.futures.as_completed(futures):
#             print("saved " + future.result())


def dataSplitGenerator(data, seed, trainP, validateP, testP, allImages, allBookLabels):
    totalItems = imageCount(data)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    rints = random.sample(range(0, totalItems), totalItems)
    trainCount, validateCount, testCount = splitNumber(trainP, validateP, testP, totalItems)

    #   Arrays with data copied to rints index values
    rImages = []
    rBookLabels = []
    for i in range(0, len(allImages)):
        randomIndex = rints[i]
        rImages.append(allImages[randomIndex])
        rBookLabels.append(allBookLabels[randomIndex])

    if testP != 0:
        trainImgItems = rImages[:trainCount]
        #     print("trainImgItems = [0->"+ str(trainCount) + "]")
        validateImgItems = rImages[trainCount:(trainCount + validateCount)]
        #     print("validateImgItems = ["+ str(trainCount) + "->" + str((trainCount+validateCount)) + "]")
        testImgItems = rImages[trainCount + validateCount:]
        #     print("testImgItems = [" + str(trainCount+validateCount) + "->" + str(totalItems-1) + "]")

        trainTxtItems = rBookLabels[:trainCount]
        validateTxtItems = rBookLabels[trainCount:(trainCount + validateCount)]
        testTxtItems = rBookLabels[trainCount + validateCount:]

    if testP == 0:
        trainImgItems = rImages[:trainCount]
        validateImgItems = rImages[trainCount:]
        trainTxtItems = rBookLabels[:trainCount]
        validateTxtItems = rBookLabels[trainCount:]
        testImgItems = []
        testTxtItems = []

    return trainImgItems, validateImgItems, testImgItems, trainTxtItems, validateTxtItems, testTxtItems


def cleanData(target):
    shutil.rmtree(target)
    print(os.path.dirname(target))
    os.makedirs(os.path.dirname(target), exist_ok=True)


def errorCheck(data, trainImgItems, validateImgItems, testImgItems, trainTxtItems, validateTxtItems, testTxtItems):
    data = readAllBooks()
    img_count = imageCount(data)

    trainfilenames = []
    validfilenames = []
    testfilenames = []
    for i in range(0, len(trainImgItems)):
        trainfilenames.append(trainImgItems[i][1])
    for i in range(0, len(validateImgItems)):
        validfilenames.append(validateImgItems[i][1])
    for i in range(0, len(testImgItems)):
        testfilenames.append(testImgItems[i][1])
    allfilenames = []
    allfilenames.extend(trainfilenames)
    allfilenames.extend(validfilenames)
    allfilenames.extend(testfilenames)

    # print(allfilenames)
    duplicate_count = 0
    skipped_count = 0
    for i in range(0, img_count):
        img_filename = "im" + str(i) + ".jpg"
        duplicates = allfilenames.count(img_filename)
        if duplicates > 1:
            duplicate_count += duplicates
            # print("%d duplicates found of %s" % (duplicates, img_filename))
        if img_filename not in allfilenames:
            skipped_count += 1
            # print("%s was skipped" %(img_filename))

    traintxt = []
    validtxt = []
    testtxt = []
    for i in range(0, len(trainTxtItems)):
        traintxt.append(trainTxtItems[i][0])
    for i in range(0, len(validateTxtItems)):
        validtxt.append(validateTxtItems[i][0])
    for i in range(0, len(testTxtItems)):
        testtxt.append(testTxtItems[i][0])
    alltxt = []
    alltxt.extend(traintxt)
    alltxt.extend(validtxt)
    alltxt.extend(testtxt)

    skips = skippedPages(data)
    totalPages = imageCount(data)
    print("skips|totalPages", len(skips), totalPages)
    print(len(skips) / totalPages * 100, "% of Manga109 pages have no corresponding text file because of missing data")
    print((totalPages-len(alltxt))/totalPages * 100, "% of yolo dataset images have no corresponding text file because of missing data")

    print("skipped | all img_count ", skipped_count, img_count)
    print((skipped_count / img_count) * 100, "% of all yolo dataset output images are skipped!")
    print("duplicate_count | len(allfilenames) ", duplicate_count, len(allfilenames))
    print((duplicate_count/len(allfilenames))*100, "% of all yolo dataset output images are duplicates!")


def saveSplit(trainImgItems, validateImgItems, testImgItems, trainTxtItems, validateTxtItems, testTxtItems):
    shutil.rmtree("Manga109")
    os.makedirs("Manga109", exist_ok=True)
    if len(trainImgItems) != 0:
        print("Saving training data")
        os.makedirs("Manga109/images/train/", exist_ok=True)
        os.makedirs("Manga109/labels/train/", exist_ok=True)
        saveFormattedImg(trainImgItems, "Manga109/images/train/")
        saveFormattedTxt(trainTxtItems, "Manga109/labels/train/")
    if len(validateImgItems) != 0:
        print("Saving validation data")
        os.makedirs("Manga109/images/val/", exist_ok=True)
        os.makedirs("Manga109/labels/val/", exist_ok=True)
        saveFormattedImg(validateImgItems, "Manga109/images/val/")
        saveFormattedTxt(validateTxtItems, "Manga109/labels/val/")
    if len(testImgItems) != 0:
        print("Saving test data")
        os.makedirs("Manga109/images/test/", exist_ok=True)
        os.makedirs("Manga109/labels/test/", exist_ok=True)
        saveFormattedImg(testImgItems, "Manga109/images/test/")
        saveFormattedTxt(testTxtItems, "Manga109/labels/test/")


if __name__ == "__main__":
    print("Starting Manga109 data processing")
    readStart = time.time()
    data = readAllBooks()
    s1 = time.time() - readStart
    print(f"\nFinished reading data, Time taken: {time.time() - readStart}\n")

    print("Starting Manga109 data formatting")
    formatStart = time.time()
    img, txt = outputPaths(data)
    s2 = time.time() - formatStart
    print(f"\nFinished formatting data, Time taken: {time.time() - formatStart}\n")

    print("Starting Manga109 data splitting")
    splitStart = time.time()
    trainImgItems, validateImgItems, testImgItems, trainTxtItems, validateTxtItems, testTxtItems = dataSplitGenerator(
        data, 109, 70, 20, 10, img, txt)
    s3 = time.time() - splitStart
    print(f"\nFinished splitting formatted data, Time taken: {time.time() - splitStart}\n")

    errorCheck(data, trainImgItems, validateImgItems, testImgItems, trainTxtItems, validateTxtItems, testTxtItems)
    print("Saving train/validate data...")
    saveStart = time.time()
    saveSplit(trainImgItems, validateImgItems, testImgItems, trainTxtItems, validateTxtItems, testTxtItems)
    s4 = time.time() - saveStart
    print(f"\nFinished saving data to folders, Time taken: {time.time() - saveStart}\n")
    total = time.time() - readStart
    print(f"\nTotal time taken: {time.time() - readStart}\n")

    a = s1 / total
    b = s2 / total
    c = s3 / total
    d = s4 / total
    print("reading%:", a * 100)
    print("formatting%:", b * 100)
    print("splitting%:", c * 100)
    print("saving%:", d * 100)