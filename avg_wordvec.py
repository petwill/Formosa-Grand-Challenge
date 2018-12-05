# -*- coding: utf-8 -*-
import jieba
import pickle
import pandas as pd
import numpy as np
import argparse
jieba.set_dictionary('./SMN_Daikon/data/dict.txt.big')
dim = 512

option_num = 6
dict_path1 = './word_embeddings/cna.cbow.cwe_p.tar_g.512d.0.txt'
dict_path2 = './word_embeddings/cna.cbow.512d.0.txt'
dict_path3 = './word_embeddings/cna.cbow.cwe_p.512d.0.txt'
dict_paths = [dict_path1]

dft_problem_path = './AIFirst_test_problem.txt'
dft_vanilla_path = './RNN-encoding-master/vanilla_weights.pickle'
dft_smn_path = './smn_weights.pickle'
dft_ans_path = './ensemble.csv'
dft_test_num = 500
parser = argparse.ArgumentParser(description='predict by avg_wordvec and ensemble with SMN and Vanilla.')
parser.add_argument('--problem_path',type=str,
					default=dft_problem_path,
					help='location of testing problems.')
parser.add_argument('--vanilla_path',type=str,
					default=dft_vanilla_path,
					help='location of vanilla weights.')
parser.add_argument('--smn_path',type=str,
					default=dft_smn_path,
					help='location of smn weights.')
parser.add_argument('--ans_path',type=str,
					default=dft_ans_path,
					help='path to ouput the answer.')
parser.add_argument('--test_num',default=dft_test_num,type=int,help='num of test problems.')
args = parser.parse_args()

def get_weights(path):
	file = open(path, 'rb')
	RNN_pred = pickle.load(file)
	file.close()
	weights = np.asarray(RNN_pred)
	return weights

def get_avg_pred():
	def get_ans(dialogue, answers,word_vecs):
		def get_avg_emb(sentence):
			emb_cnt = 0
			avg_emb = np.zeros((dim,))
			sentence = sentence.strip('\n').strip(',')

			for word in jieba.cut(sentence):
				if word in word_vecs:
						avg_emb += word_vecs[word]#*item.weight
						emb_cnt += 1
			if emb_cnt!=0:
				avg_emb /= emb_cnt
			return avg_emb

		avg_dlg_emb = get_avg_emb(dialogue)

		# 在六個回答中，每個答句都取詞向量平均作為向量表示
		# 我們選出與dialogue句子向量表示cosine similarity最高的短句
		scores=np.zeros([6,])
		for idx, ans in enumerate(answers):
			avg_ans_emb = get_avg_emb(ans)
			if np.linalg.norm(avg_dlg_emb)!=0 and np.linalg.norm(avg_ans_emb)!=0:
				sim = np.dot(avg_dlg_emb, avg_ans_emb) / np.linalg.norm(avg_dlg_emb) / np.linalg.norm(avg_ans_emb)
			else:
				sim=0.
			scores[idx] = sim
		return scores

	def wordvecs_list():
		def get_emb_dict(path):
			word_vecs = {}
			# 開啟詞向量檔案
			with open(path, encoding='utf-8') as f:
				for line in f:
					# 假設我們的詞向量有300維
					# 由word以及向量中的元素共301個
					# 以空格分隔組成詞向量檔案中一行
					tokens = line.strip().split()

					# 第一行是兩個整數，分別代表有幾個詞向量，以及詞向量維度
					if len(tokens) == 2:
						dim = int(tokens[1])
						continue

					word = tokens[0]
					vec = np.array([float(t) for t in tokens[1:]])
					word_vecs[word] = vec
			return word_vecs
		word_vecs_list=[]
		for path in dict_paths:
			word_vecs_list.append(get_emb_dict(path))
		return word_vecs_list

	word_vecs_list = wordvecs_list()
	df = pd.read_csv(args.problem_path, index_col=[0], sep=',')
	score_array = np.zeros((len(word_vecs_list),args.test_num,option_num),dtype=float)
	for idx_word_vecs,word_vecs in enumerate(word_vecs_list):
		for id in range(1,args.test_num+1):
			dialogue = df['dialogue'][id]
			answers = df['options'][id].split('\t')
			score_array[idx_word_vecs,id-1] = get_ans(dialogue, answers,word_vecs)
	ans_array = np.zeros([args.test_num,option_num],dtype=float)
	for i in range(len(word_vecs_list)):
		ans_array += score_array[i]
	return ans_array

def write_ans(ans_array, path):
	with open(path, 'w+') as file:
		file.write('id,ans\n')
		for id, ans in enumerate(ans_array):
			file.write(str(id+1)+','+str(ans)+'\n')

if __name__=='__main__':
	avg_pred = get_avg_pred()
	vanilla_pred = get_weights(args.vanilla_path)
	smn_pred = get_weights(args.smn_path)

	ans_array = avg_pred + vanilla_pred + smn_pred
	ans_array = np.argmax(ans_array,1)

	write_ans(ans_array, args.ans_path)

