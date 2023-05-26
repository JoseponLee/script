from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import numpy as np
import csv
from evaluate import load

def indexMany(s,str):
    length = len(s)
    str1 = s
    list = []
    sum = 0
    try:
        while str1.index(str)!=-1:
            n = str1.index(str)
            str2 = str1[0:n + 1]
            str1 = str1[n + 1:length]
            sum = sum + len(str2)
            list.append(sum - 1)
            length=length-len(str2)
    except ValueError:
        return list
    return list









def wup(word1, word2, alpha):
    """
    calculate the wup similarity
    :param word1:
    :param word2:
    :param alpha:
    :return:
    """
    # print(word1, word2)
    if word1 == word2:
        return 1.0

    w1 = wordnet.synsets(word1)
    w1_len = len(w1)
    if w1_len == 0: return 0.0
    w2 = wordnet.synsets(word2)
    w2_len = len(w2)
    if w2_len == 0: return 0.0

    #match the first
    word_sim = w1[0].wup_similarity(w2[0])
    if word_sim is None:
        word_sim = 0.0

    if word_sim < alpha:
        word_sim = 0.1*word_sim
    return word_sim


def wups(words1, words2, alpha):
    """

    :param pred:
    :param truth:
    :param alpha:
    :return:
    """
    sim = 1.0
    flag = False
    for w1 in words1:
        max_sim = 0
        for w2 in words2:
            word_sim = wup(w1, w2, alpha)
            if word_sim > max_sim:
                max_sim = word_sim
        if max_sim == 0: continue
        sim *= max_sim
        flag = True
    if not flag:
        sim = 0.0
    return sim


def get_wups(pred, truth, alpha):
    """
    calculate the wups score
    :param pred:
    :param truth:
    :return:
    """
    pred = word_tokenize(pred)
    truth = word_tokenize(truth)
    item1 = wups(pred, truth, alpha)
    item2 = wups(truth, pred, alpha)
    value = min(item1, item2)
    return value


def bert_score(prediction, reference):
    result = \
    bertscore.compute(predictions=[prediction], references=[reference], model_type="microsoft/deberta-xlarge-mnli")[
        'precision']
    # print(result, prediction, reference)
    return result[0]



if __name__ == '__main__':
    splits = ['train']
    abandon_list = ['why', 'does', 'did', 'what', 'the', 'do', 'are', 'at', 'is']
    bertscore = load("bertscore")



    for split in splits:
        file_name = 'after_amt_after_MC_process_how_after_sort_' + split + '.csv'
        out_path = 'after_amt_after_MC_process_how_after_sort_after_pos_order_byaction_bert0.75_0.5' + split + '.csv'
        action_dict = {}
        action_dict_ans = {}
        pair_dict_pos = {}
        pair_dict_neg = {}
        pair_dict_line = {}
        csv_reader = csv.reader(open(file_name))
        line_list = []
        id = 0
        for line in csv_reader:
            if line[0] != 'video_id':
                id += 1
                # line[13] action, line[14] lemma. line[15] lemma id
                Q, A, action = line[4], line[int(line[5]) + 8], line[13]
                line.append(id)
                line_list.append(line)
                if action not in action_dict.keys():
                    action_dict[action] = [(id, line)]
                    action_dict_ans[action] = [A]
                else:
                    action_dict[action].append((id, line))
                    action_dict_ans[action].append(A)


        for action in action_dict_ans.keys():
            for i, A in enumerate(action_dict_ans[action]):
                for j, A_pair in enumerate(action_dict_ans[action][i+1:]):

                    # value_wups = get_wups(A, A_pair, 0)
                    value = bert_score(A, A_pair)

                    id1, id2 = action_dict[action][i][0], action_dict[action][i + j + 1][0]
                    if id1 not in pair_dict_line.keys():
                        pair_dict_line[id1] = action_dict[action][i][1]
                    if id2 not in pair_dict_line.keys():
                        pair_dict_line[id2] = action_dict[action][i + j + 1][1]
                    # if value_wups>=0.85 and value<0.7:
                    #     print(value_wups, value, action_dict_ans[action][i], '|||', action_dict_ans[action][j + i + 1], id1, id2)
                    if value >= 0.75:
                        print(value, action_dict_ans[action][i], '|||', action_dict_ans[action][j + i + 1], id1, id2)
                        if id1 not in pair_dict_pos.keys():
                            pair_dict_pos[id1] = [id2]
                        else:
                            pair_dict_pos[id1].append(id2)

                        if id2 not in pair_dict_pos.keys():
                            pair_dict_pos[id2] = [id1]
                        else:
                            pair_dict_pos[id2].append(id1)
                    elif value < 0.5:
                        if id1 not in pair_dict_neg.keys():
                            pair_dict_neg[id1] = [id2]
                        else:
                            pair_dict_neg[id1].append(id2)

                        if id2 not in pair_dict_neg.keys():
                            pair_dict_neg[id2] = [id1]
                        else:
                            pair_dict_neg[id2].append(id1)




        with open(out_path, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["video_id", "frame_count", "width", "height", "question", "answer", "qid", "type", "a0", "a1", "a2", "a3", "a4", "action", "lemma", "lemma_id", "id", "pos_id", "neg_id"])
            for line in line_list:
                id = line[-1]
                if id not in pair_dict_pos.keys():
                    line.append('')
                else:
                    line.append(process_id_list(pair_dict_pos[id]))
                if id not in pair_dict_neg.keys():
                    line.append('')
                else:
                    line.append(process_id_list(pair_dict_neg[id]))
                writer.writerow(line)




