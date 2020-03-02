from konlpy.tag import Kkma
import json
import os
import collections

pos_tagger = Kkma()

INPUT_FILENAME = "Data\\result.json"
TOKENIZED_RESULT_FILENAME = "Data\\tokenizedResult.txt"
INDEXED_DICTIONARY_FILENAME = "Data\\IndexDictionary.txt"
INDEXED_TOKENIZED_ASSESSMENT_RESULT_FILENAME = "Data\\IndexedTokenizedAssessmentResult.txt"



def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.nouns(doc)]


def read_raw_data(filename):
    """
    Amount of Assessment are 11055

    :param filename: location of result.json
    :return:
    """
    with open(filename, 'r', encoding='utf-8') as f:
        decode = json.loads(f.readline())
        return decode


def build_lectureAssessmentList(decodedJson):
    """

    :param decodedJson:
    :return:
    """
    lectureAssessmentList = list()

    for i, lecturetList in enumerate(decodedJson):
        # decodeDataList include each of lectures

        for j, lectureAssessment in enumerate(lecturetList['articles']):
            # each of lectures has lecture Assessment
            lectureAssessmentList.append(lectureAssessment)

    return lectureAssessmentList


def write_tokenizedAssessment(filename, tokenizedAssessmentList):
    """

    :param tokenizedAssessmentList: AssessmentList through Kkma tokenizer
    :return: none
    """

    with open(filename, 'a', encoding='utf-8') as f:
        f.write(tokenizedAssessmentList[0] + "\t" + str(tokenizedAssessmentList[1]) + "\n")


def get_word_index(wordList, indexDictionary):
    indexed_Assessment = list()
    for word in wordList:
        if word in indexDictionary:
            # print(word, indexDictionary[word])
            indexed_Assessment.append(indexDictionary[word])
        else:
            indexed_Assessment.append('0')

    return indexed_Assessment


def build_indexed_tokenized_result_file(indexDictionary):

    indexed_Assessment_List = list()

    rate_List = list()

    with open(TOKENIZED_RESULT_FILENAME, 'r', encoding='utf-8') as f:
        tokenizedResult = f.read()
        lines = tokenizedResult.split('\n')
        for i, line in enumerate(lines):
            assessment, rate = line.split('\t')

            sequence_segment = (get_word_index(assessment.split(' '), indexDictionary))[:80]
            padding = ['0'] * (80 - len(sequence_segment))
            sequence_segment = sequence_segment + padding


            indexed_Assessment_List.append(','.join(sequence_segment))
            rate_List.append(rate)


    return indexed_Assessment_List, rate_List


def write_indexed_tokenized_result_file(filename, indexed_Assessment_List, rate_List):
    with open(filename, 'w', encoding='utf-8') as f:
        for data in list(zip(indexed_Assessment_List, rate_List)):
            """
            if int(data[1]) <= 2:
                label = 1
            elif int(data[1]) == 3:
                label = 2
            elif int(data[1]) > 3:
                label = 3
            """
            if int(data[1])>=3:
                label = 2
            else:
                label = 1

            f.write(data[0] + "," + str(label) + "\n")


if __name__ == '__main__':

    if (os.path.exists(TOKENIZED_RESULT_FILENAME) == False):

        decodedJson = read_raw_data(INPUT_FILENAME)
        lectureAssessmentList = build_lectureAssessmentList(decodedJson=decodedJson)

        del decodedJson

        tokenizedAssessmentList = list()

        for i, lectureAssessment in enumerate(lectureAssessmentList):
            # 이유 알수없는 process finished error 때문에 아래처럼 하나 토크나이징하고 바로 파일에 작성하기를 반복함
            write_tokenizedAssessment(TOKENIZED_RESULT_FILENAME, tokenizedAssessmentList=[
                ' '.join(tokenize(lectureAssessmentList[i]['text'])).replace('/', ''),
                lectureAssessmentList[i]['rate']])


    elif (os.path.exists(INDEXED_TOKENIZED_ASSESSMENT_RESULT_FILENAME) == False ):
        indexDictionary = dict()

        # Read Index Dictionary pre made by trainW2V.py
        with open(INDEXED_DICTIONARY_FILENAME, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                data = line.split("\t")
                indexDictionary[data[1].replace('\n', '')] = data[0]  # line is data form of [(index, word)]

        indexed_Assessment_List, rate_List = build_indexed_tokenized_result_file(
            indexDictionary=indexDictionary)
        write_indexed_tokenized_result_file(filename=INDEXED_TOKENIZED_ASSESSMENT_RESULT_FILENAME,
                                            indexed_Assessment_List=indexed_Assessment_List,
                                            rate_List=rate_List)
