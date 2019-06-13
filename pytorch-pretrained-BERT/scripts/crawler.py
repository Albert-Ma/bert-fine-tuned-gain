import nltk
import time
import random
import argparse
from tqdm import tqdm

from bs4 import BeautifulSoup
from selenium import webdriver
from bs4 import NavigableString
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

BASE_URL = 'https://fanyi.baidu.com/?aldtype=16047#en/zh/'


# def load_html(url=None):
#     """Open web from browser."""
#     try:
#         browser = webdriver.Chrome()
#         browser.get(url)
#     except:
#         return None
#     return browser
#
#
# def parse_html(brownser, word, flag=True):
#     """Parse web and invoked once for synonyms.
#     :return [{original word: sentence_list},{synonyms word: sentence_list}]"""
#     origin = {}
#     result = []
#     synonyms_click = 'side-nav'
#     nav = brownser.find_element_by_class_name(synonyms_click)
#     a = nav.find_element_by_class_name('nav-item')
#     WebDriverWait(brownser, 10).until(
#         EC.element_to_be_clickable((By.CLASS_NAME, synonyms_click))).click()
#     exit(0)
#     # origin[word] = crawler(brownser, biligual_examples_xpath)
#     # print(origin)
#     # if this word does not have biliguai examples,
#     if origin[word] is None:
#         brownser.quit()
#         return {}
#     # or this word does not have a verbed sentence
#     origin = check_word_pos(origin)
#     if len(origin[word]) == 0:
#         brownser.quit()
#         return {}
#     if flag:
#         result.append(origin)
#         synonyms_xpath = '//*[@id="synonyms"]/ul'
#         try: # check if this word has `synonyms`
#             brownser.find_element_by_xpath(synonyms_click)
#             WebDriverWait(brownser, 10).until(
#             EC.element_to_be_clickable((By.XPATH, synonyms_click))).click()
#             synonyms = brownser.find_element_by_xpath(synonyms_xpath)
#         except:
#             print("word ({}) has no synonyms".format(word))
#             brownser.quit()
#             return result
#         word_group = synonyms.find_elements_by_class_name('search-js')
#         # '#synonyms > ul > p:nth-child(2) > span:nth-child(1) > a'
#         for w in word_group:
#             try: # if there is something wrong here, we skip it.
#                 driver = load_html(w.get_attribute('href'))
#                 if driver is None:
#                     driver.quit()
#                     time.sleep(60)
#                     continue
#                 # We just consider one single word.
#                 if len(w.text.strip().split()) > 1:
#                     driver.quit()
#                     continue
#                 sysn = parse_html(driver, w.text, False)
#                 print("word ({})'s sysn ({})".format(word, sysn))
#                 if len(sysn) == 0 or len(sysn[w.text]) == 0:
#                     driver.quit()
#                 else:
#                     result.append(sysn)
#                     driver.quit()
#             except:
#                 continue
#         brownser.quit()
#         return result
#     else:
#         # if this is the last web to crawl, we return dict{word: sentences}
#         brownser.quit()
#         return origin
#
#
# def crawler(brownser, xpath):
#     """Crawl data from web.
#     :return sentence_list"""
#     original = []
#     try:
#         brownser.find_element_by_xpath(xpath)
#     except:
#         return None
#     soup = BeautifulSoup(brownser.page_source, "html.parser")
#     res = soup.select('#bilingual > ul')
#     for sents in res:    # ul
#         for s in sents:   # <li>
#             sents = []
#             for i, p in enumerate(s):
#                 if isinstance(p, str):
#                     continue
#                 else:
#                     if i == 1:
#                         if len(p) != 0:
#                             for sp in p:
#                                 if isinstance(sp, NavigableString):
#                                     continue
#                                 sents.append(sp.text.strip())
#             if len(sents) == 0:
#                 continue
#             original.append(" ".join(sents))
#     return original
#
#
# def build_pairwise(result):
#     """Build pairwise data."""
#     # word1 \t word2 \t sentence1 \t sentence2
#     assert len(result) > 1
#     lines = []
#     for i, item in enumerate(result):
#         word = list(item)[0]
#         sentence_list = list(item.values())[0]
#         if len(sentence_list) == 0:
#             print("original word:{} does not have a 'verbed' sentence"
#                   .format(word))
#             return lines
#         # print(word, sentence_list)
#         if i == 0:
#             origin_word = word
#             origin_sentences = sentence_list
#         else:
#             for sentence in sentence_list:
#                 lines.append("{}\t{}\t{}\t{}".
#                              format(origin_word, word,
#                                     random.sample(origin_sentences, 1)[0],
#                                     sentence))
#     return lines
#
#
# def check_word_pos(word_sentence_dict):
#     """Check if this word is a VB, delete those which it's not a VB sentence.
#     :return {word: sentence_list}"""
#     word = list(word_sentence_dict)[0]
#     sentences = list(word_sentence_dict.values())[0]
#
#     result_sentences = []
#     pos_tags = []
#     for sentence in sentences:
#         # TODO: cause we only use one single word, so we do a word_tokenize and do one word match
#         pos_tags.append(nltk.pos_tag(nltk.word_tokenize(sentence)))
#     for i, pos_tag_sentence in enumerate(pos_tags):
#         # sentence
#         flag = False
#         for pos_tag_word in pos_tag_sentence:
#             if pos_tag_word[0] == word:
#                 if str(pos_tag_word[1]).startswith('VB'):
#                     print("word:({}), in this sentence:({}) is a verb."
#                           .format(word, sentences[i]))
#                     flag = True
#                     break
#         if flag:
#             result_sentences.append(sentences[i])
#     return {word: result_sentences}
#
#
# def main():
#     words = []
#     with open(args.vocab_file, 'r') as f:
#         lines = f.readlines()
#         for i, word in enumerate(lines):
#             if i <= 647:
#                 continue
#             if i > 2000:
#                 break
#             words.append(word.split('\t')[0])
#
#     words = ['make']
#     c = 0
#     with open(args.output_file, 'a+') as writer:
#         for i, word in enumerate(tqdm(words)):
#             # if c > 647:
#             #     print("lines:{} break.".format(i))
#             #     break
#             driver = load_html(BASE_URL+word)
#             result = parse_html(driver, word)
#             # if this word does not have syns
#             if len(result) <= 1:
#                 continue
#             lines = build_pairwise(result)
#             print("word:{}, result:{}".format(word, lines))
#             print("*"*20)
#             for line in lines:
#                 if len(line.strip()) == 0:
#                     continue
#                 writer.write(line + '\n')
#                 c += 1
#                 # writer.close()
#                 # exit(0)
#             driver.quit()
#             # time.sleep(5)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--vocab_file", type=str, required=True)
#     parser.add_argument("--output_file", default='result.txt', type=str, required=True)
#
#     args = parser.parse_args()
#
#     main()


import execjs
import requests
import re
import json

JS_CODE = """
function a(r, o) {
    for (var t = 0; t < o.length - 2; t += 3) {
        var a = o.charAt(t + 2);
        a = a >= "a" ? a.charCodeAt(0) - 87 : Number(a),
        a = "+" === o.charAt(t + 1) ? r >>> a: r << a,
        r = "+" === o.charAt(t) ? r + a & 4294967295 : r ^ a
    }
    return r
}
var C = null;
var token = function(r, _gtk) {
    var o = r.length;
    o > 30 && (r = "" + r.substr(0, 10) + r.substr(Math.floor(o / 2) - 5, 10) + r.substring(r.length, r.length - 10));
    var t = void 0,
    t = null !== C ? C: (C = _gtk || "") || "";
    for (var e = t.split("."), h = Number(e[0]) || 0, i = Number(e[1]) || 0, d = [], f = 0, g = 0; g < r.length; g++) {
        var m = r.charCodeAt(g);
        128 > m ? d[f++] = m: (2048 > m ? d[f++] = m >> 6 | 192 : (55296 === (64512 & m) && g + 1 < r.length && 56320 === (64512 & r.charCodeAt(g + 1)) ? (m = 65536 + ((1023 & m) << 10) + (1023 & r.charCodeAt(++g)), d[f++] = m >> 18 | 240, d[f++] = m >> 12 & 63 | 128) : d[f++] = m >> 12 | 224, d[f++] = m >> 6 & 63 | 128), d[f++] = 63 & m | 128)
    }
    for (var S = h,
    u = "+-a^+6",
    l = "+-3^+b+-f",
    s = 0; s < d.length; s++) S += d[s],
    S = a(S, u);

    return S = a(S, l),
    S ^= i,
    0 > S && (S = (2147483647 & S) + 2147483648),
    S %= 1e6,
    S.toString() + "." + (S ^ h)
}
"""


class Dict:
    def __init__(self):
        self.sess = requests.Session()
        self.headers = {
            'User-Agent':
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'
        }
        self.token = None
        self.gtk = None

        # 获得token和gtk
        # 必须要加载两次保证token是最新的，否则会出现998的错误
        self.loadMainPage()
        self.loadMainPage()

    def loadMainPage(self):
        """
            load main page : https://fanyi.baidu.com/
            and get token, gtk
        """
        url = 'https://fanyi.baidu.com'

        try:
            r = self.sess.get(url, headers=self.headers)
            self.token = re.findall(r"token: '(.*?)',", r.text)[0]
            self.gtk = re.findall(r"window.gtk = '(.*?)';", r.text)[0]
        except Exception as e:
            raise e
            # print(e)

    def langdetect(self, query):
        """
            post query to https://fanyi.baidu.com/langdetect
            return json
            {"error":0,"msg":"success","lan":"en"}
        """
        url = 'https://fanyi.baidu.com/langdetect'
        data = {'query': query}
        try:
            r = self.sess.post(url=url, data=data)
        except Exception as e:
            raise e
            # print(e)

        json = r.json()
        if 'msg' in json and json['msg'] == 'success':
            return json['lan']
        return None

    def dictionary(self, query):
        """
            max query count = 2
            get translate result from https://fanyi.baidu.com/v2transapi
        """
        url = 'https://fanyi.baidu.com/v2transapi'

        sign = execjs.compile(JS_CODE).call('token', query, self.gtk)

        lang = self.langdetect(query)
        data = {
            'from': 'en' if lang == 'en' else 'zh',
            'to': 'zh' if lang == 'en' else 'en',
            'query': query,
            'simple_means_flag': 3,
            'sign': sign,
            'token': self.token,
        }
        try:
            r = self.sess.post(url=url, data=data)
        except Exception as e:
            raise e

        if r.status_code == 200:
            json = r.json()
            if 'error' in json:
                raise Exception('baidu sdk error: {}'.format(json['error']))
                # 998错误则意味需要重新加载主页获取新的token
            return json
        return None

    def dictionary_by_lang(self, query, fromlang, tolang):
        """
            max query count = 2
            get translate result from https://fanyi.baidu.com/v2transapi
        """
        url = 'https://fanyi.baidu.com/v2transapi'

        sign = execjs.compile(JS_CODE).call('token', query, self.gtk)

        lang = self.langdetect(query)
        data = {
            'from': fromlang,
            'to': tolang,
            'query': query,
            'simple_means_flag': 3,
            'sign': sign,
            'token': self.token,
        }
        try:
            r = self.sess.post(url=url, data=data)
        except Exception as e:
            raise e

        if r.status_code == 200:
            json = r.json()
            if 'error' in json:
                raise Exception('baidu sdk error: {}'.format(json['error']))
                # 998错误则意味需要重新加载主页获取新的token
            # print(json)
            return self.parse_data(json)
        return None

    def trans_baidu_en1(self, text):
        the_ret = self.dictionary_by_lang(text, "zh", "en")
        ret1 = self.dictionary_by_lang(the_ret, "en", "zh")
        return ret1

    def parse_data(self, json):
        synonym_data = json["dict_result"]
        # check if this word have synonyms
        if 'synonym' in synonym_data:
            synonym_data = synonym_data["synonym"]
        else:
            return None
        pairwise_result = []
        for item in synonym_data:
            # 'words' are not always correct
            # words = item['words']
            # TODO: a (an)
            synonyms = item['synonyms']
            synonyms_list = []
            words = []
            for item in synonyms:
                if 'ex' in item and len(item['ex']) != 0:
                    synonyms_list.append(item['ex'])
                    words.append(item['syn']['word'])
                elif 'be' in item: # 'after'
                    tmp = []
                    for i, sub_item in enumerate(item['be']['item']):
                        tmp.append(sub_item['ex'])
                    # print(tmp)
                    synonyms_list.append(tmp[0])
                    words.append(item['syn']['word'])
                else:
                    raise ValueError("word do not have 'ex'")
            # print(synonyms_list)
            for i in range(len(words)-1):
                for j in range(i+1, len(words)):
                    line = "{}\t{}\t{}\t{}"\
                        .format(words[i], words[j],
                                random.sample(synonyms_list[i], 1)[0]['enText'],
                                random.sample(synonyms_list[j], 1)[0]['enText'])
                    pairwise_result.append(line)
        return pairwise_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", type=str, required=True)
    parser.add_argument("--output_file", default='result.txt', type=str, required=True)
    parser.add_argument("--min_count", default=100, type=int)
    parser.add_argument("--debug", action='store_true')

    args = parser.parse_args()

    baidu_dict = Dict()
    words = []
    with open(args.vocab_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i < 788:
                continue
            word, count = line.split('\t')
            if int(count) > args.min_count:
                words.append(word)
    if args.debug:
        words = ['after', 'too', 'speed']
    synonyms_vocab = set()
    with open(args.output_file, 'a+') as writer:
        for i, word in enumerate(tqdm(words)):
            res = baidu_dict.dictionary_by_lang(word, "en", "zh")
            print("="*20)
            print("word:{}, result:{}".format(word, res))
            # word do not have synonyms
            if res is None:
                continue
            for line in res:
                if len(line.strip()) == 0:
                    continue
                word_a, word_b, _, _ = line.split('\t')
                if '\t'.join([word_a, word_b]) not in synonyms_vocab:
                    synonyms_vocab.add('\t'.join([word_a, word_b]))
                    writer.write(line + '\n')
            time.sleep(1)
