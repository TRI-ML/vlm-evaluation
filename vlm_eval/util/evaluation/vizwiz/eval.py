# coding=utf-8

__author__ = "Qing Li"

# This code is based on the code written by Qing Li for VizWiz Python API available at the following link:
# (https://github.com/xxx)


import copy
import re
import sys

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


class VQAEval:
    def __init__(self, gts, res, n=2):
        self.n = n
        self.accuracy = {}
        self.caption_metric = {}
        self.evalQA = {}
        self.gts = gts
        self.res = res
        self.unanswerability = {}
        self.original_gts = copy.deepcopy(self.gts)
        self.original_res = copy.deepcopy(self.res)
        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn`t`ve",
            "couldnt`ve": "couldn`t`ve",
            "didnt": "didn`t",
            "doesnt": "doesn`t",
            "dont": "don`t",
            "hadnt": "hadn`t",
            "hadnt`ve": "hadn`t`ve",
            "hadn'tve": "hadn`t`ve",
            "hasnt": "hasn`t",
            "havent": "haven`t",
            "hed": "he`d",
            "hed`ve": "he`d`ve",
            "he`dve": "he`d`ve",
            "hes": "he`s",
            "howd": "how`d",
            "howll": "how`ll",
            "hows": "how`s",
            "Id`ve": "I`d`ve",
            "I`dve": "I`d`ve",
            "Im": "I`m",
            "Ive": "I`ve",
            "isnt": "isn`t",
            "itd": "it`d",
            "itd`ve": "it`d`ve",
            "it`dve": "it`d`ve",
            "itll": "it`ll",
            "let`s": "let`s",
            "maam": "ma`am",
            "mightnt": "mightn`t",
            "mightnt`ve": "mightn`t`ve",
            "mightn`tve": "mightn`t`ve",
            "mightve": "might`ve",
            "mustnt": "mustn`t",
            "mustve": "must`ve",
            "neednt": "needn`t",
            "notve": "not`ve",
            "oclock": "o`clock",
            "oughtnt": "oughtn`t",
            "ow`s`at": "`ow`s`at",
            "`ows`at": "`ow`s`at",
            "`ow`sat": "`ow`s`at",
            "shant": "shan`t",
            "shed`ve": "she`d`ve",
            "she`dve": "she`d`ve",
            "she`s": "she`s",
            "shouldve": "should`ve",
            "shouldnt": "shouldn`t",
            "shouldnt`ve": "shouldn`t`ve",
            "shouldn`tve": "shouldn`t`ve",
            "somebody`d": "somebodyd",
            "somebodyd`ve": "somebody`d`ve",
            "somebody`dve": "somebody`d`ve",
            "somebodyll": "somebody`ll",
            "somebodys": "somebody`s",
            "someoned": "someone`d",
            "someoned`ve": "someone`d`ve",
            "someone`dve": "someone`d`ve",
            "someonell": "someone`ll",
            "someones": "someone`s",
            "somethingd": "something`d",
            "somethingd`ve": "something`d`ve",
            "something`dve": "something`d`ve",
            "somethingll": "something`ll",
            "thats": "that`s",
            "thered": "there`d",
            "thered`ve": "there`d`ve",
            "there`dve": "there`d`ve",
            "therere": "there`re",
            "theres": "there`s",
            "theyd": "they`d",
            "theyd`ve": "they`d`ve",
            "they`dve": "they`d`ve",
            "theyll": "they`ll",
            "theyre": "they`re",
            "theyve": "they`ve",
            "twas": "`twas",
            "wasnt": "wasn`t",
            "wed`ve": "we`d`ve",
            "we`dve": "we`d`ve",
            "weve": "we've",
            "werent": "weren`t",
            "whatll": "what`ll",
            "whatre": "what`re",
            "whats": "what`s",
            "whatve": "what`ve",
            "whens": "when`s",
            "whered": "where`d",
            "wheres": "where's",
            "whereve": "where`ve",
            "whod": "who`d",
            "whod`ve": "who`d`ve",
            "who`dve": "who`d`ve",
            "wholl": "who`ll",
            "whos": "who`s",
            "whove": "who've",
            "whyll": "why`ll",
            "whyre": "why`re",
            "whys": "why`s",
            "wont": "won`t",
            "wouldve": "would`ve",
            "wouldnt": "wouldn`t",
            "wouldnt`ve": "wouldn`t`ve",
            "wouldn`tve": "wouldn`t`ve",
            "yall": "y`all",
            "yall`ll": "y`all`ll",
            "y`allll": "y`all`ll",
            "yall`d`ve": "y`all`d`ve",
            "y`alld`ve": "y`all`d`ve",
            "y`all`dve": "y`all`d`ve",
            "youd": "you`d",
            "youd`ve": "you`d`ve",
            "you`dve": "you`d`ve",
            "youll": "you`ll",
            "youre": "you`re",
            "youve": "you`ve",
        }
        self.manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        self.articles = ["a", "an", "the"]

        self.periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile(r"(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def evaluate(self):
        # =================================================
        # Compute accuracy
        # =================================================
        accQA = []
        print("computing accuracy")
        step = 0
        for img in self.gts.keys():
            resAns = self.res[img]["answer"]
            resAns = resAns.replace("\n", " ")
            resAns = resAns.replace("\t", " ")
            resAns = resAns.strip()
            resAns = self.processPunctuation(resAns)
            resAns = self.processDigitArticle(resAns)
            gtAcc = []

            for i, _ans in enumerate(self.gts[img]["answers"]):
                otherGTAns = [item for j, item in enumerate(self.gts[img]["answers"]) if i != j]
                matchingAns = [item for item in otherGTAns if item["answer"] == resAns]
                acc = min(1, float(len(matchingAns)) / 3)
                gtAcc.append(acc)

            avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
            accQA.append(avgGTAcc)

            self.setEvalQA(img, avgGTAcc)
            if step % 100 == 0:
                self.updateProgress(step / float(len(self.gts.keys())))
            step = step + 1

        self.setAccuracy(accQA)
        print("Done computing accuracy")

    def get_answerable_preds(self):
        # Determine whether the VLM predicted answerable or not
        for img in self.res.keys():
            resAns = self.res[img]["answer"]
            resAns = resAns.replace("\n", " ")
            resAns = resAns.replace("\t", " ")
            resAns = resAns.strip()
            resAns = self.processPunctuation(resAns)
            resAns = self.processDigitArticle(resAns)
            self.original_res[img]["answerable"] = int(resAns != "unanswerable")

    def evaluate_unanswerability(self):
        self.get_answerable_preds()
        pred = []
        gt_labels = []
        for img in self.gts.keys():
            gt_labels.append(self.original_gts[img]["answerable"])
            pred.append(self.original_res[img]["answerable"])
        gt_labels = np.array(gt_labels)
        pred = np.array(pred)

        gt_labels_n = 1 - gt_labels
        pred_n = 1.0 - pred
        average_precision = average_precision_score(gt_labels_n, pred_n)
        one_f1_score = f1_score(gt_labels_n, pred_n > 0.5)

        self.unanswerability["average_precision"] = round(100 * average_precision, self.n)
        self.unanswerability["f1_score"] = round(100 * one_f1_score, self.n)

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (re.search(self.commaStrip, inText) is not None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText

    def setAccuracy(self, accQA):
        self.accuracy["overall"] = round(100 * float(sum(accQA)) / len(accQA), self.n)

    def setEvalQA(self, img, acc):
        self.evalQA[img] = round(100 * acc, self.n)

    def updateProgress(self, progress):
        barLength = 20
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength * progress))
        text = "\rFinshed Percent: [{0}] {1}% {2}".format(
            "#" * block + "-" * (barLength - block), int(progress * 100), status
        )
        sys.stdout.write(text)
        sys.stdout.flush()
