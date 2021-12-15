import os
from distutils.dir_util import copy_tree

fin_metriclist = open('supported_metrics.txt', 'r')
metriclist = set([row.strip() for row in fin_metriclist.readlines()])

fin_algolist = open('supported_algorithms.txt', 'r')
algolist = set([row.strip() for row in fin_algolist.readlines()])
print(algolist)

fin_template = open('./template/config.xml', 'r')
template_xml = fin_template.read()
# print(template_xml)

for algo in ['efm']:
    # 'wbpr', 'bpr', 'plsa', 'biasedmf', 'mostpopular', 'randomguess', 'itemknn', 'userknn', 'tfidf'
    for metric in ['auc', 'rmse']:
        # 'ndcg', 'precision', 'recall', 
        new_xml = template_xml.replace('randomguess', algo)
        new_xml = new_xml.replace('rmse', metric)
        if metric in ['auc', 'precision', 'recall']:
            new_xml = new_xml.replace('boolean', 'true')
        elif metric == 'ndcg':
            new_xml = new_xml.replace('boolean', 'true')
            new_xml = new_xml.replace('<!-- <list-size>10</list-size> -->', '<list-size>10</list-size>')
        else:
            new_xml = new_xml.replace('boolean', 'false')
        # print(new_xml)
        try:
            os.mkdir('./'+algo)
        except:
            pass
        copy_tree('./data/', './'+algo+'/data/')
        try:
            os.mkdir('./'+algo+'/'+metric)
        except:
            pass
        try:
            os.mkdir('./'+algo+'/'+metric+'/conf/')
        except Exception as e:
            print(e)
        fout = open('./'+algo+'/'+metric+'/conf/config.xml', 'w')
        fout.write(new_xml)
        fout.close()
        