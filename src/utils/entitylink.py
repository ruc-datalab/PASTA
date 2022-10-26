# The following script is adapted from the script of TabFact.
# Original: https://github.com/wenhuchen/Table-Fact-Checking/blob/5ea13b8f6faf11557eec728c5f132534e7a22bf7/code/preprocess_data.py

import sys
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import pandas
from nltk.tokenize import WordPunctTokenizer
import json
import re
import time
import string
from unidecode import unidecode
from multiprocessing import Pool
import multiprocessing
import time

with open('/datasets/tabfact/data/freq_list.json') as f:
    vocab = json.load(f)

with open('/datasets/tabfact/data/stop_words.json') as f:
    stop_words = json.load(f)

months_a = ['january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december']
months_b = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
a2b = {a: b for a, b in zip(months_a, months_b)}
b2a = {b: a for a, b in zip(months_a, months_b)}


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def augment(s):
    recover_dict = {}
    if 'first' in s:
        s.append("1st")
        recover_dict[s[-1]] = 'first'
    elif 'second' in s:
        s.append("2nd")
        recover_dict[s[-1]] = 'second'
    elif 'third' in s:
        s.append("3rd")
        recover_dict[s[-1]] = 'third'
        s.append("3")
        recover_dict[s[-1]] = 'third'
    elif 'fourth' in s:
        s.append("4th")
        recover_dict[s[-1]] = 'fourth'
    elif 'fifth' in s:
        s.append("5th")
        recover_dict[s[-1]] = 'fifth'
    elif 'sixth' in s:
        s.append("6th")
        recover_dict[s[-1]] = 'sixth'

    for i in range(1, 10):
        if "0" + str(i) in s:
            s.append(str(i))
            recover_dict[s[-1]] = "0" + str(i)

    if 'crowd' in s or 'attendance' in s:
        s.append("people")
        recover_dict[s[-1]] = 'crowd'
        s.append("audience")
        recover_dict[s[-1]] = 'crowd'

    if "Never-smoker" in s:
        s.append("Non-smoker")
        recover_dict[s[-1]] = "Never-smoker"
        s.append('non-smoker')
        recover_dict[s[-1]] = 'Never-smoker'

    if 'Smoking' in s and 'status' in s and "Never" in s:
        s.append('Non-smoker')
        recover_dict[s[-1]] = 'Smoking status : Never'
        s.append('non-smoker')
        recover_dict[s[-1]] = 'Smoking status : Never'
        #print("in never:",s)

    if 'Smoking' in s and 'status' in s and 'Current' in s:
        s.append('Heavy')
        recover_dict[s[-1]] = 'Smoking status : Current' 
        s.append('heavy')
        recover_dict[s[-1]] = 'Smoking status : Current' 
        # s.append('smoker')
        # recover_dict[s[-1]] = 'Smoking' 
        #print("in current:",s)
    if 'Smoking' in s and 'Yes' in s:
        s.append('Heavy')
        recover_dict[s[-1]] = 'Smoking Yes'
        s.append('heavy')
        recover_dict[s[-1]] = 'Smoking Yes'
    if 'Smoking' in s and 'No' in s:
        s.append('non-smoker')
        recover_dict[s[-1]] = 'Smoking No'
        s.append('Non-smoker')
        recover_dict[s[-1]] = 'Smoking No'
    if 'Active' in s and 'smoker' in s:
        s.append("Heavy")
        recover_dict[s[-1]] = 'Active'
        s.append("heavy")
        recover_dict[s[-1]] = 'Active'
    if 'Never' in s and 'smoker' in s:
        s.append("non-smoker")
        recover_dict[s[-1]] = 'Never smoker'
        s.append("Non-smoker")
        recover_dict[s[-1]] = 'Never smoker'


    if any([_ in months_a + months_b for _ in s]):
        for i in range(1, 32):
            if str(i) in s:
                if i % 10 == 1:
                    s.append(str(i) + "st")
                elif i % 10 == 2:
                    s.append(str(i) + "nd")
                elif i % 10 == 3:
                    s.append(str(i) + "rd")
                else:
                    s.append(str(i) + "th")
                recover_dict[s[-1]] = str(i)

        for k in a2b:
            if k in s:
                s.append(a2b[k])
                recover_dict[s[-1]] = k

        for k in b2a:
            if k in s:
                s.append(b2a[k])
                recover_dict[s[-1]] = k

    return s, recover_dict


def replace_useless(s):
    s = s.replace(',', '')
    s = s.replace('.', '')
    s = s.replace('/', '')
    return s


def get_closest(inp, string, indexes, tabs, threshold):

    if string in stop_words:
        return None

    dist = 10000
    rep_string = replace_useless(string)
    len_string = len(rep_string.split())

    minimum = []
    for index in indexes:
        entity = replace_useless(tabs[index[0]][index[1]])
        len_tab = len(entity.split())
        if abs(len_tab - len_string) < dist:
            minimum = [index]
            dist = abs(len_tab - len_string)
        elif abs(len_tab - len_string) == dist:
            minimum.append(index)

    vocabs = []
    for s in rep_string.split(' '):
        vocabs.append(vocab.get(s, 10000))

    # Whether contain rare words
    if dist == 0:
        return minimum[0]

    # String Length
    feature = [len_string]
    # Proportion
    feature.append(-dist / (len_string + dist + 0.) * 4)
    if any([(s.isdigit() and int(s) < 100) for s in rep_string.split()]):
        feature.extend([0, 0])
    else:
        # Quite rare words
        if max(vocabs) > 1000:
            feature.append(1)
        else:
            feature.append(-1)
        # Whether contain super rare words
        if max(vocabs) > 5000:
            feature.append(3)
        else:
            feature.append(0)
    # Whether it is only a word
    if len_string > 1:
        feature.append(1)
    else:
        feature.append(0)
    # Whether candidate has only one
    if len(indexes) == 1:
        feature.append(1)
    else:
        feature.append(0)
    # Whether cover over half of it
    if len_string > dist:
        feature.append(1)
    else:
        feature.append(0)

    # Whether contains alternative
    cand = replace_useless(tabs[minimum[0][0]][minimum[0][1]])
    if '(' in cand and ')' in cand:
        feature.append(2)
    else:
        feature.append(0)
    # Match more with the header
    if minimum[0][0] == 0:
        feature.append(2)
    else:
        feature.append(0)
    # Whether it is a month
    if any([" " + _ + " " in " " + rep_string + " " for _ in months_a + months_b]):
        feature.append(5)
    else:
        feature.append(0)

    # Whether it matches against the candidate
    
    if rep_string in cand:
        feature.append(0)
    else:
        feature.append(-5)

    #if string=="Heavy" or string=="Non-smoker":
    if(string=="Heavy" or string=="heavy") and cand=="Smoking status : Current":
        return minimum[0]
    if (string=="Non-smoker" or string=="non-smoker") and cand=="Smoking status : Never":
        return minimum[0]
    if (string=="Non-smoker" or string=="non-smoker") and cand=="Smoking No":
        return minimum[0]
    if (string=="Heavy" or string=="heavy") and cand=="Smoking Yes":
        return minimum[0]
    if (string=="Heavy" or string=="heavy") and cand=="Active smoker":
        return minimum[0]
    if (string=="Non-smoker" or string=="non-smoker") and cand=="Never smoker":
        return minimum[0]

    if sum(feature) > threshold:
        if len(minimum) > 1:
            if minimum[0][0] > 0:
                return minimum[0] #[-2, minimum[0][1]]
            else:
                return minimum[0]
        else:
            return minimum[0]
    else:
        return None


def replace_number(string):
    string = re.sub(r'(\b)one(\b)', r'\g<1>1\g<2>', string)
    string = re.sub(r'(\b)two(\b)', '\g<1>2\g<2>', string)
    string = re.sub(r'(\b)three(\b)', '\g<1>3\g<2>', string)
    string = re.sub(r'(\b)four(\b)', '\g<1>4\g<2>', string)
    string = re.sub(r'(\b)five(\b)', '\g<1>5\g<2>', string)
    string = re.sub(r'(\b)six(\b)', '\g<1>6\g<2>', string)
    string = re.sub(r'(\b)seven(\b)', '\g<1>7\g<2>', string)
    string = re.sub(r'(\b)eight(\b)', '\g<1>8\g<2>', string)
    string = re.sub(r'(\b)nine(\b)', '\g<1>9\g<2>', string)
    string = re.sub(r'(\b)ten(\b)', '\g<1>10\g<2>', string)
    string = re.sub(r'(\b)eleven(\b)', '\g<1>11\g<2>', string)
    string = re.sub(r'(\b)twelve(\b)', '\g<1>12\g<2>', string)
    string = re.sub(r'(\b)thirteen(\b)', '\g<1>13\g<2>', string)
    string = re.sub(r'(\b)fourteen(\b)', '\g<1>14\g<2>', string)
    string = re.sub(r'(\b)fifteen(\b)', '\g<1>15\g<2>', string)
    string = re.sub(r'(\b)sixteen(\b)', '\g<1>16\g<2>', string)
    string = re.sub(r'(\b)seventeen(\b)', '\g<1>17\g<2>', string)
    string = re.sub(r'(\b)eighteen(\b)', '\g<1>18\g<2>', string)
    string = re.sub(r'(\b)nineteen(\b)', '\g<1>19\g<2>', string)
    string = re.sub(r'(\b)twenty(\b)', '\g<1>20\g<2>', string)
    return string


def replace(w, transliterate):
    if w in transliterate:
        return transliterate[w]
    else:
        return w


def intersect(w_new, w_old):
    new_set = []
    for w_1 in w_new:
        for w_2 in w_old:
            if w_1[:2] == w_2[:2] and w_1[2] > w_2[2]:
                new_set.append(w_2)
    return new_set


def recover(buf, recover_dict, content):
    if len(recover_dict) == 0:
        return buf
    else:
        new_buf = []
        for w in buf.split(' '):
            if w not in content:
                new_buf.append(recover_dict.get(w, w))
            else:
                new_buf.append(w)
        return ' '.join(new_buf)


def postprocess(inp, backbone, trans_backbone, transliterate, tabs, recover_dicts, repeat, threshold=1.0):
    new_str = []
    non_recover_str=[]
    new_tags = []
    buf = ""
    non_r_buf=""
    pos_buf = []
    last = set()
    prev_closest = []
    inp, _, pos_tags = get_lemmatize(inp, True)
    #print("inp:",inp)
    printflag=0
    for w, p in zip(inp, pos_tags):
        #print(new_str)
        if (w in backbone) and ((" " + w + " " in " " + buf + " " and w in repeat) or (" " + w + " " not in " " + buf + " ")):
            #print("1:",w)
            #print("buf:",buf)
            if buf == "":
                last = set(backbone[w])
                buf = w
                non_r_buf = w
                pos_buf.append(p)
            else:
                proposed = set(backbone[w]) & last
                #print(buf,proposed)
                if not proposed:
                    closest = get_closest(inp, buf, last, tabs, threshold)
                    if closest: 
                        non_r_buf = '#{};{},{}#'.format(buf, closest[0], closest[1])
                        buf = '#{};{},{}#'.format(recover(buf, recover_dicts[closest[0]][closest[1]],
                                                          tabs[closest[0]][closest[1]]), closest[0], closest[1])
                    elif len(last) != 0:
                        #print(last)
                        pos = list(list(last)[0])
                        buf = '#{};{},{}#'.format(buf, pos[0], pos[1])
                        
                    new_str.append(buf)
                    non_recover_str.append(non_r_buf)
                    if buf.startswith("#"):
                        new_tags.append('ENT')
                    else:
                        new_tags.extend(pos_buf)
                    pos_buf = []
                    buf = w
                    non_r_buf = w
                    last = set(backbone[w])
                    pos_buf.append(p)
                else:
                    last = proposed
                    buf += " " + w
                    non_r_buf += " " + w
                    pos_buf.append(p)

        elif w in trans_backbone and ((" " + w + " " in " " + buf + " " and w in repeat) or (" " + w + " " not in " " + buf + " ")):
            #print("2:",w)
            if buf == "":
                last = set(trans_backbone[w])
                buf = transliterate[w]
                non_r_buf = transliterate[w]
                pos_buf.append(p)
            else:
                
                proposed = set(trans_backbone[w]) & last
                if not proposed:
                    closest = get_closest(inp, buf, last, tabs, threshold)
                    if closest:
                        #print(buf)
                        non_r_buf = '#{};{},{}#'.format(buf, closest[0], closest[1])
                        buf = '#{};{},{}#'.format(recover(buf, recover_dicts[closest[0]][closest[1]],
                                                          tabs[closest[0]][closest[1]]), closest[0], closest[1])
                    else:
                        buf = '#{};{},{}#'.format(buf, trans_backbone[w][0][0], trans_backbone[w][0][1])
                    new_str.append(buf)
                    non_recover_str.append(non_r_buf)
                    if buf.startswith("#"):
                        new_tags.append('ENT')
                    else:
                        new_tags.extend(pos_buf)
                    pos_buf = []
                    buf = transliterate[w]
                    non_r_buf = transliterate[w]
                    last = set(trans_backbone[w])
                    pos_buf.append(p)
                else:
                    buf += " " + transliterate[w]
                    non_r_buf += " " + transliterate[w]
                    last = proposed
                    pos_buf.append(p)

        else:
            if buf != "":
                closest = get_closest(inp, buf, last, tabs, threshold)
                if closest:
                    non_r_buf = '#{};{},{}#'.format(buf, closest[0], closest[1])
                    buf = '#{};{},{}#'.format(recover(buf, recover_dicts[closest[0]][closest[1]],
                                                      tabs[closest[0]][closest[1]]), closest[0], closest[1])
                elif len(last) != 0:
                    pos = list(list(last)[0])
                    buf = '#{};{},{}#'.format(buf, pos[0], pos[1])
                new_str.append(buf)
                non_recover_str.append(non_r_buf)
                if buf.startswith("#"):
                    new_tags.append('ENT')
                else:
                    new_tags.extend(pos_buf)
                pos_buf = []

            buf = ""
            non_r_buf = ""
            last = set()
            new_str.append(replace_number(w))
            non_recover_str.append(replace_number(w))
            new_tags.append(p)
    if buf != "":
        closest = get_closest(inp, buf, last, tabs, threshold)
        if closest:
            non_r_buf = '#{};{},{}#'.format(buf, closest[0], closest[1])
            buf = '#{};{},{}#'.format(recover(buf, recover_dicts[closest[0]][closest[1]],
                                              tabs[closest[0]][closest[1]]), closest[0], closest[1])
        elif len(last) != 0:
            pos = list(list(last)[0])
            buf = '#{};{},{}#'.format(buf, pos[0], pos[1])
        new_str.append(buf)
        non_recover_str.append(non_r_buf)
        if buf.startswith("#"):
            new_tags.append('ENT')
        else:
            new_tags.extend(pos_buf)
        pos_buf = []

    return " ".join(new_str), " ".join(new_tags)," ".join(non_recover_str)


def get_lemmatize(words, return_pos):
    #words = nltk.word_tokenize(words)
    recover_dict = {}
    #words = words.strip().split(' ')
    words = WordPunctTokenizer().tokenize(words)
    pos_tags = [_[1] for _ in nltk.pos_tag(words)]
    word_roots = []
    for w, p in zip(words, pos_tags):
        if is_ascii(w):
            lemm = lemmatizer.lemmatize(w, get_wordnet_pos(p))
            #print(lemm)
            if lemm != w:
                recover_dict[lemm] = w
            word_roots.append(lemm)
        else:
            word_roots.append(w)
    if return_pos:
        return word_roots, recover_dict, pos_tags
    else:
        return word_roots, recover_dict


tag_dict = {"JJ": wordnet.ADJ,
            "NN": wordnet.NOUN,
            "NNS": wordnet.NOUN,
            "NNP": wordnet.NOUN,
            "NNPS": wordnet.NOUN,
            "VB": wordnet.VERB,
            "VBD": wordnet.VERB,
            "VBG": wordnet.VERB,
            "VBN": wordnet.VERB,
            "VBP": wordnet.VERB,
            "VBZ": wordnet.VERB,
            "RB": wordnet.ADV,
            "RP": wordnet.ADV}
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'

lemmatizer = WordNetLemmatizer()


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def merge_strings(name, string, tags=None):
    buff = ""
    inside = False
    words = []

    for c in string:
        if c == "#" and not inside:
            inside = True
            buff += c
        elif c == "#" and inside:
            inside = False
            buff += c
            words.append(buff)
            buff = ""
        elif c == " " and not inside:
            if buff:
                words.append(buff)
            buff = ""
        elif c == " " and inside:
            buff += c
        else:
            buff += c

    if buff:
        words.append(buff)

    tags = tags.split(' ')
    assert len(words) == len(tags), "{} and {}".format(words, tags)

    i = 0
    while i < len(words):
        if i < 2:
            i += 1
        elif words[i].startswith('#') and (not words[i - 1].startswith('#')) and words[i - 2].startswith('#'):
            if is_number(words[i].split(';')[0][1:]) and is_number(words[i - 2].split(';')[0][1:]):
                i += 1
            else:
                prev_idx = words[i - 2].split(';')[1][:-1].split(',')
                cur_idx = words[i].split(';')[1][:-1].split(',')
                if cur_idx == prev_idx or (prev_idx[0] == '-2' and prev_idx[1] == cur_idx[1]):
                    position = "{},{}".format(cur_idx[0], cur_idx[1])
                    candidate = words[i - 2].split(';')[0] + " " + words[i].split(';')[0][1:] + ";" + position + "#"
                    words[i] = candidate
                    del words[i - 1]
                    del tags[i - 1]
                    i -= 1
                    del words[i - 1]
                    del tags[i - 1]
                    i -= 1
                else:
                    i += 1
        else:
            i += 1

    return " ".join(words), " ".join(tags)


def sub_func(fnm, caption, claim):
    #name, entry = inputs
    #print("name:",name)
    #print('entry:',entry)
    backbone = {}
    trans_backbone = {}
    transliterate = {}
    tabs = []
    recover_dicts = []
    repeat = set()
    rindex=0
    result_list = {}
    with open('/datasets/tabfact/data/all_csv/' + fnm, 'r') as f:
        for k, _ in enumerate(f.readlines()):#row
            #_ = _.decode('utf8')
            tabs.append([])
            recover_dicts.append([])
            for l, w in enumerate(_.strip().split('#')):#col
                #w = w.lower()
                #print(l,w)
                if (k==0):
                    if(l==0):
                        result_list["thead"]=[w]
                    else:
                        result_list["thead"].append(w)
                elif k==1:
                    if(l==0):
                        result_list["tbody"]=[[w]]
                    else:
                        result_list["tbody"][k-1].append(w)
                else:
                    if(l==0):
                        result_list["tbody"].append([w])
                    else:
                        result_list["tbody"][k-1].append(w)
                    
                #w = w.replace(',', '').replace('  ', ' ')
                #w=w.lower()
                tabs[-1].append(w)
                tabs[-1].append(w.lower())
                if len(w) > 0:
                    lemmatized_w, recover_dict = get_lemmatize(w, False)
                    lemmatized_w, new_dict = augment(lemmatized_w)
                    recover_dict.update(new_dict)
                    recover_dicts[-1].append(recover_dict)
                    for i, sub in enumerate(lemmatized_w):
                        if sub not in backbone:
                            backbone[sub] = [(k, l)]

                            if not is_ascii(sub):
                                trans_backbone[unidecode(sub)] = [(k, l)]
                                transliterate[unidecode(sub)] = sub
                        else:
                            if (k, l) not in backbone[sub]:

                                backbone[sub].append((k, l))
                            else:
                                if sub not in months_a + months_b:
                                    repeat.add(sub)
                            if not is_ascii(sub):
                                trans_backbone[unidecode(sub)].append((k, l))
                                transliterate[unidecode(sub)] = sub

                    for i, sub in enumerate(w.split(' ')):
                        if sub not in backbone:
                            backbone[sub] = [(k, l)]

                            if not is_ascii(sub):
                                trans_backbone[unidecode(sub)] = [(k, l)]
                                transliterate[unidecode(sub)] = sub
                        else:
                            if (k, l) not in backbone[sub]:                  
                                backbone[sub].append((k, l))
   
                            if not is_ascii(sub):
                                trans_backbone[unidecode(sub)].append((k, l))
                                transliterate[unidecode(sub)] = sub
                else:
                    recover_dicts[-1].append({})
                    #raise ValueError("Empty Cell")

    # Masking the caption
    captions, _ = get_lemmatize(caption.strip(), False)
    for i, w in enumerate(captions):
        if w not in backbone:
            backbone[w] = [(-1, -1)]
        else:
            backbone[w].append((-1, -1))
    tabs.append([" ".join(captions)])
    #print("tabs:",tabs)
    #print("backbone:",backbone)
    backbone_temp = backbone.copy()
    for i in backbone_temp:
        lower_w = i.lower()
        if lower_w != i:
            backbone[lower_w] = backbone[i]
    if "avg" in backbone:
        backbone["average"] = backbone["avg"]
    if "weight" in backbone:
        backbone["weigh"] = backbone["weight"]
    elif "weigh" in backbone:
        backbone["weight"] = backbone["weigh"]
    if "height" in backbone:
        backbone["high"] = backbone["height"]
        backbone["tall"] = backbone["height"]
    elif "high" in backbone:
        backbone["height"] = backbone["high"]
        backbone["tall"] = backbone["high"]
    if "subsidiary" in backbone:
        backbone["subsidy"] = backbone["subsidiary"]
    if "point" in backbone and "score" not in backbone:
        backbone['score'] = backbone["point"]
    elif "score" in backbone and "point" not in backbone:
        backbone['point'] = backbone["score"]
    if "win" in backbone:
        backbone["won"] = backbone["win"]
    elif "won" in backbone:
        backbone["win"] = backbone["won"]
    if "loss" in backbone and "lost" not in backbone:
        backbone['lost'] = backbone['loss']
    elif "lost" in backbone and "loss" not in backbone:
        backbone['loss'] = backbone['lost']
    if "enrollment" in backbone:
        backbone['enrol'] = backbone['enrollment']
    if "attendance" in backbone:
        backbone['attendee'] = backbone['attendance']
        backbone['attended'] = backbone['attendance']
    if 'pos' in backbone:
        backbone['position'] = backbone['pos']
    if 'visitor' in backbone:
        backbone['visit'] = backbone['visitor']
        backbone['visiting'] = backbone['visitor']

    results = []
    recover=[]

    orig_sent = claim
    if "=" not in orig_sent:
        sent, tags, rec = postprocess(orig_sent, backbone, trans_backbone,
                                    transliterate, tabs, recover_dicts, repeat, threshold=1.0)
        
        if "#" not in sent:
            sent, tags,rec = postprocess(orig_sent, backbone, trans_backbone,
                                        transliterate, tabs, recover_dicts, repeat, threshold=0.0)
        #print(sent)
        return sent






def get_func(filename, output,result_list):
    with open(filename) as f:
        data = json.load(f)
    r1_results = {}
    names = []
    entries = []
    for name in data:
        names.append(name)
        entries.append(data[name])

    t1 = time.time()
    r=[]
    rec=[]
    for input in zip(names,entries):
        na,res,result_list,recover=sub_func(input,result_list)
        r.append((na,res))
        rec.append((na,recover))
    return dict(r),result_list,dict(rec)
