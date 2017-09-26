# -*- coding: utf-8 -*-
import sys, os
# import io, os, sys
# import numpy
import operator
import collections
import time

from nltk import PorterStemmer
porter_stemmer=PorterStemmer()

do_stemming = False

RUN_SAVE = 1
RUN_EVAL = 2
running_type = RUN_SAVE

mod_name = sys.argv[0]
run_type = sys.argv[1]
doc_path = sys.argv[2]

doc_name = os.path.basename(doc_path)
doc_dir = os.path.dirname(doc_path)

meta_dir = './meta'
if not os.path.exists(meta_dir):
    os.makedirs(meta_dir)

known_words_path = meta_dir + '/known_words.txt'

def truncate_word(word):
	start = 0
	while start < len(word) and word[start].isalpha() == False:
		start += 1
	end = len(word)
	while end > start and word[end-1].isalpha() == False:
		end -= 1
	truncated = word[start:end].lower()
	for letter in truncated:
		if letter.isalpha():
			break
	else:
		return ''
	if truncated.find('http://') == 0:
		return ''
	if do_stemming == True:
		if len(truncated) == 0:
			return ''
	else:
		return truncated

def tokenization(text):
	if len(text) == 0:
		return
	start_pos = 0
	while start_pos < len(text):
		while start_pos < len(text):
			if text[start_pos].isalpha():
				break
			else:
				start_pos += 1
		end_pos = start_pos
		while end_pos < len(text):
			if text[end_pos].isalpha():
				end_pos += 1
			else:
				break
		word = text[start_pos:end_pos].lower()
		if word.find('urgent') == -1:
			yield text[start_pos:end_pos]
		start_pos = end_pos

def read_txt(text):
	bag_words = collections.OrderedDict()
	for word in tokenization(text):
		truncated = truncate_word(word)
		if truncated != '':
			try:
				bag_words[truncated] += 1
			except KeyError:
				bag_words[truncated] = 1
	return bag_words

def stem_read_txt(text):
	bag_words = collections.OrderedDict()
	for word in tokenization(text):
		truncated = truncate_word(word)
		truncated = porter_stemmer.stem(truncated)

		if truncated != '':
			try:
				bag_words[truncated] += 1
			except KeyError:
				bag_words[truncated] = 1
	return bag_words

def refine_doc(path):
	
	doc = []
	
	try:
		with open(path, 'r', encoding='UTF8') as f:
			lines = f.readlines()
			for line in lines:
				line = line.strip().lower()
				if len(line) > 0:
					doc.append(line)
	
	except FileNotFoundError as fe:
		print(fe)
		doc = []

	return doc

def save_doc(doc, path):

	with open(path, 'a', encoding='UTF8') as f:
		for line in doc:
			f.write(line + '\n')

	return

def create_bag_words(doc):

	print('Creating bag of words...')
	ts = time.time()

	bag_words = collections.OrderedDict()
	bag_words_stemmed = collections.OrderedDict()

	for line in doc:
		disasembled = read_txt(line)
		disasembled_stemmed = stem_read_txt(line)

		for stemmed in disasembled_stemmed:
			if stemmed not in bag_words_stemmed:
				bag_words_stemmed[stemmed] = collections.OrderedDict()

		
		for word in disasembled:
			stemmed = porter_stemmer.stem(word)
			if word not in bag_words_stemmed[stemmed]:
				bag_words_stemmed[stemmed][word] = disasembled[word]
			else:
				bag_words_stemmed[stemmed][word] += disasembled[word]

			try:
				bag_words[stemmed] += disasembled[word]
			except KeyError:
				bag_words[stemmed] = disasembled[word]

	te = time.time() - ts
	print('Done: Creating bag of words. Execution time: %.3fs' % te)

	return bag_words, bag_words_stemmed

def get_key(item):
	return int(item['word'])

def read_known_words():

	print('Loading known words...')
	ts = time.time()

	known_words = collections.OrderedDict()

	try:
		with open(known_words_path, 'r', encoding='UTF8') as f:
			for word in f:
				word = word.strip().lower()
				stemmed_word = porter_stemmer.stem(word)
				try:
					known_words[stemmed_word] += 1
				except KeyError:
					known_words[stemmed_word] = 1
	except FileNotFoundError as fe:
		pass

	te = time.time() - ts
	print('Done: Loading known words. Execution time: %.3fs' % te)

	return known_words

def is_known_word(known_words_stemmed, word):

	stemmed_word = porter_stemmer.stem(word)
	
	return stemmed_word in known_words_stemmed

def add_known_word(word):
	try:
		with open(known_words_path, 'a', encoding='UTF8') as f:
			f.write(word + '\n')
	except FileNotFoundError as fe:
		pass	

def save_voca(bag_words, bag_words_stemmed, known_words):
	
	print('Saving vocabulary files...')
	ts = time.time()

	sorted_voca = sorted(bag_words.items(), key=operator.itemgetter(1), reverse=True)
	
	voca_path = doc_dir+'/voca'
	voca_hash_path = doc_dir+'/voca_hash'
	if os.path.exists(voca_path):
		os.remove(voca_path)
	if os.path.exists(voca_hash_path):
		os.remove(voca_hash_path)

	voca_file = open(voca_path, 'a', encoding='UTF8')
	voca_hash_file = open(voca_hash_path, 'a', encoding='UTF8')
	

	for word in sorted_voca:
		# print('%s\t%d' % (word[0], word[1]))

		if is_known_word(known_words, word[0]) == True:
			continue

		voca_file.write(str(word[0]) + '\t' + str(word[1]) + '\n')
		for each in sorted(bag_words_stemmed[word[0]], key=bag_words_stemmed[word[0]].get, reverse=True):
			voca_hash_file.write(word[0] + '\t' + each + '\t' + str(bag_words_stemmed[word[0]][each]) + '\n')

	voca_file.close()
	voca_hash_file.close()

	te = time.time() - ts
	print('Done: Saving vocabulary files. Execution time: %.3fs' % te)

	return 1

def eval_voca(bag_words, bag_words_stemmed, known_words):
	
	print('Evaluate vocabulary ...')
	ts = time.time()

	sorted_voca = sorted(bag_words.items(), key=operator.itemgetter(1), reverse=True)
	new_known = collections.OrderedDict()

	for word in sorted_voca:
		# print('%s\t%d' % (word[0], word[1]))

		if is_known_word(known_words, word[0]) == True:
			continue

		res = input('Do you know %s ?[Y/n] ' % (list(bag_words_stemmed[word[0]])[0]))
		res = res.strip().lower()
		# if res == '' or res == 'y' or res == 'yes':
		if res == '' or res == 'y' or res == 'yes':
			try:
				new_known[word] += 1
			except KeyError:
				new_known[word] = 1
		if res == 'q' or res == 'quit':
			break

	for item in new_known:
		add_known_word(item[0])

	te = time.time() - ts
	print('Done: Evaluate vocabulary. Execution time: %.3fs' % te)

	return 1

def get_input_type(input_type):

	res = RUN_SAVE
	input_type = input_type.strip().lower()
	if input_type == '-s' or input_type == 'save':
		res = RUN_SAVE
	if input_type == '-e' or input_type == 'eval':
		res = RUN_EVAL

	return res

# ****************************************************
# Do tasks

running_type = get_input_type(run_type)

if running_type == RUN_SAVE:
	title = 'Creating vocabulary'
else:
	title = 'Evaluate vocabulary'
print('%s ...' % (title))
time_start = time.time()

# Refine docs
doc = refine_doc(doc_path)

# Stemming and creating bag of words
bag_words, bag_words_stemmed = create_bag_words(doc)

# Read already known words
known_words = read_known_words()

# Save vocabulary files
if running_type == RUN_SAVE:
	res = save_voca(bag_words, bag_words_stemmed, known_words)
else:
	res = eval_voca(bag_words, bag_words_stemmed, known_words)


time_elapsed = time.time() - time_start
print('Done: %s. Execution time: %.3fs' % (title, time_elapsed))












