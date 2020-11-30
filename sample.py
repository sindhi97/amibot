from flask import Flask, render_template, request, json, url_for, redirect
from flask_jsglue import JSGlue
# from SPARQLWrapper import SPARQLWrapper, JSON
#from ColProperty import ColProperty
from flask_cors import CORS, cross_origin
from flask import jsonify
from werkzeug import secure_filename
import json, ast
import pickle
import urllib
import os
from pathlib import Path

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
#------ 
# from nltk.corpus import stopwords
# stopWords = set(stopwords.words('english'))
import re
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk.corpus import wordnet

# -----

import numpy as np
import tflearn
import tensorflow as tf
# import pywrap_tensorflow
from tensorflow import pywrap_tensorflow
import random
import pickle
import xlrd
from flask import send_from_directory

import sys
from os import walk

directory='./PYTHON_FILE'
global domain_module
for (dirpath, dirnames, filenames) in walk(directory):
    for filename in filenames:
        fnparts = filename.split('.')
        if len(fnparts)>1 and fnparts[-1] == 'py':
            sys.path.append(directory)
            print('@@',fnparts[0])
            domain_module = __import__(fnparts[0])
    break
# from doctor import *


UPLOAD_FOLDER = 'inputFiles'
UPLOAD_xl='xl_files'
UPLOAD_PY='PYTHON_FILE'
ALLOWED_EXTENSIONS = set(['json','xls','py'])

if not os.path.exists("inputFiles"):
	os.mkdir("inputFiles")
if not os.path.exists("xl_files"):
	os.mkdir("xl_files")
if not os.path.exists("PYTHON_FILE"):
	os.mkdir("PYTHON_FILE")
sample = Flask(__name__)
sample.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
sample.config['UPLOAD_xl'] = UPLOAD_xl
sample.config['UPLOAD_PY']=UPLOAD_PY

jsglue = JSGlue(sample)

@sample.route("/")
def main():
	return render_template('ksrc_ui.html')

@sample.route("/open_info_enter/", methods=['POST'])
def open_info_enter():
#Moving forward code
	user_name = request.form['name']
	password=request.form['password']
	if user_name=="admin" and password=="123":
		return render_template('info_enter.html')
	else:
		return render_template('ksrc_ui.html')

@sample.route("/open_chat_front/", methods=['POST'])
def open_chat_front():
#Moving forward code
	global words, classes, intents, model 
	words, classes, intents, model = res_model()
	domain_module.initialise()

	json_con={}
	json_con[userID]=''
	with open('context_info.json',mode='w') as json_data:
		json.dump(json_con,json_data)


	print('=====================Yash ki tasalli--------')
	# global doc_words,doc_classes,allsymptoms,doc_model
	# doc_words,doc_classes,allsymptoms,doc_model = load_doctor()
	return render_template('chat_front.html')



@sample.route("/saveJsonFile",methods=['POST'])
def saveJsonFile():
	print(request.get_data())
	file_url=request.get_data().split(";")[0]
	filename=request.get_data().split(";")[1]
	print("file_url---------->",file_url)
	print("filename-------->",filename)
	# #pickle.dump( file, open( filename, "wb" ) )
	# # link = "http://www.somesite.com/details.pl?urn=2344"
	f = urllib.urlopen(file_url)
	file = f.read()
	print(file)
	pickle.dump( file, open( filename, "wb" ) )
	return "success"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# Template hyperlink Sumitesh - 27 April
@sample.route('/uploads/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    uploads = os.path.join(sample.root_path, sample.config['UPLOAD_FOLDER'])
    return send_from_directory(directory='download_template', filename=filename)

##########END###############################

# Updated by Sumitesh --12 Apr-2018########################

@sample.route('/uploaded', methods=['GET','POST'])#this '/' will be the root ie it will display the home play of the website
def uploaded():
	global filenames
	filenames=[]
	global filename
	filename=''
	#print (">>>>>>>>",os.getcwd())
	if request.method == 'POST':  
	  print ("----->>>>>>>>>",request.files.getlist("uploadedfiles[]"))
	  uploaded_files = request.files.getlist("uploadedfiles[]")

	  print("---------->",uploaded_files)
	  for file in uploaded_files:
	    if file and allowed_file(file.filename):

	        filename = secure_filename(file.filename)
	        print("))))))))))",('./'+os.path.join(sample.config['UPLOAD_FOLDER'])+'/'+filename))
	        ext=filename.split(".")[1]
	        if ext=="xls":
	            file.save(os.path.join(sample.config['UPLOAD_xl'], filename))
	            os.rename(os.path.join(sample.config['UPLOAD_xl'], filename), os.path.join(sample.config['UPLOAD_xl'], 'reception.xls'))
	            filenames.append(filename)
	            filename = 'reception.xls'
	            xlfilename = './'+os.path.join(sample.config['UPLOAD_xl'])+'/'+filename
	            
	            filename = filename.split(".")[0]+'.json'
	            jsonfilename = './'+os.path.join(sample.config['UPLOAD_FOLDER'])+'/'+filename
	            # print('$$$$$$$$$$$$$$$$$$--',xlfilename,jsonfilename)
	            xlToJsonConverter(xlfilename, jsonfilename)
	            
	        else:
	            file.save(os.path.join(sample.config['UPLOAD_FOLDER'], filename))
	            filenames.append(filename)


	  #return redirect(url_for('uploaded_file', filename=filename))
	filenames=set(filenames)
	filenames=list(filenames)
	print("########",filename)
	# ext=filename.split(".")[1]
	# if ext=="xls":
	#     jsonfilename = './'+os.path.join(sample.config['UPLOAD_FOLDER'])+'/'+filename
	#     xlToJsonConverter(filename, jsonfilename)

	return json.dumps(filenames)

@sample.route('/upload_py', methods=['GET','POST'])#this '/' will be the root ie it will display the home play of the website
def upload_py():
	global filenames	
	filenames=[]
	global filename
	filename=''
	#print (">>>>>>>>",os.getcwd())
	if request.method == 'POST':  
	  print ("----->>>>>>>>>",request.files.getlist("uploadedpyfiles[]"))
	  uploaded_files = request.files.getlist("uploadedpyfiles[]")

	  print("---------->",uploaded_files)
	  for file in uploaded_files:
	    if file and allowed_file(file.filename):

	        filename = secure_filename(file.filename)
	        print("))))))))))",('./'+os.path.join(sample.config['UPLOAD_PY'])+'/'+filename))
	        ext=filename.split(".")[1]
	        file.save(os.path.join(sample.config['UPLOAD_PY'], filename))
	        filenames.append(filename)


	  #return redirect(url_for('uploaded_file', filename=filename))
	filenames=set(filenames)
	filenames=list(filenames)
	print("########",filename)
	# ext=filename.split(".")[1]
	# if ext=="xls":
	#     jsonfilename = './'+os.path.join(sample.config['UPLOAD_FOLDER'])+'/'+filename
	#     xlToJsonConverter(filename, jsonfilename)

	return json.dumps(filenames)


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                # if show_details:
                #     print ("found in bag: %s" % w)
    
    return(np.array(bag))

def file_is_empty(path):		#to check if a file is empty
    return os.stat(path).st_size==0

def classify(sentence):
    ERROR_THRESHOLD = 0.25
    # generate probabilities from the model
#     print (model.predict([bow(sentence, words)])[0])
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    #print(results)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

# context={}
userID='Context'
def response(sentence, show_details=False):
    infofile="./context_info.json"
    my_file = Path(infofile)
    if my_file.is_file() and (not file_is_empty(infofile)):
        with open(infofile) as json_data:
            context = json.load(json_data)
    else:
        context = {}
        context[userID]=''

    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    print("----",results)
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    if ((not 'context_filter' in i) and context[userID]!='') or ('context_filter' in i and i['context_filter'] != context[userID]):
                        continue
                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                            (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]): 
                        if 'functioncall' in i:
                            func = i['functioncall']
                            try:
                            	func_call = getattr(domain_module, func)
                            	ret_str, context[userID] = func_call(sentence)
                            except AttributeError:
                            	ret_str = 'Sorry, I did not get it.'
                            	context[userID] = ''
                            # ret_str, context[userID] = globals()[str(func)](sentence)
                        else:
                            # a random response from the intent
                            ret_str = random.choice(i['responses'])
                            # set context for this intent if necessary
                            if 'context_set' in i:
                                context[userID] = i['context_set']
                        with open(infofile,mode='w') as json_data:
                            json.dump(context,json_data)
                        return ret_str
            results.pop(0)
    # else:
    for i in intents['intents']:
    	# print('~~~',context)
    	# print('###',i)
    	# print('@@@',context[userID])
        if (userID in context and 'notag_context' in i and i['notag_context'] == context[userID]):
            if not 'context_filter' in i or \
                    (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                if 'functioncall' in i:
                    func = i['functioncall']
                    func_call = getattr(domain_module, func)
                    ret_str, context[userID] = func_call(sentence)
                    # ret_str,context[userID] = globals()[str(func)](sentence)
                else:
                    ret_str = random.choice(i['responses'])
                # set context for this intent if necessary
                    if 'context_set' in i:
                        context[userID] = i['context_set']
                with open(infofile,mode='w') as json_data:
                    json.dump(context,json_data)
                return ret_str
    with open(infofile,mode='w') as json_data:
        json.dump(context,json_data)


def identify_pos(String):
    #String = 'Ravana was killed in a war'

    Sentences = nltk.sent_tokenize(String)
    Tokens = []
    for Sent in Sentences:
        Tokens.append(nltk.word_tokenize(Sent)) 
    Words_List = [nltk.pos_tag(Token) for Token in Tokens]

    Nouns_List = []
    #print (Words_List)
    for List in Words_List:
        for Word in List:
            if re.match('[JJ.*]', Word[1]) or re.match('[NN.*]', Word[1]):
                 Nouns_List.append(Word[0])

    Names = []
    for Nouns in Nouns_List:
        if not wordnet.synsets(Nouns):
            Names.append(Nouns)

    #print (Names)
    names = ''.join(Names)
    return names

# #########################################################



@sample.route("/train_chatbot",methods=['POST'])
def train_chatbot():
	with open('inputFiles/'+filename) as json_data:
		intents = json.load(json_data)
		words = []
		classes = []
		documents = []
		ignore_words = ['?']
		# loop through each sentence in our intents patterns
		for intent in intents['intents']:
		    #print (intent)
		    for pattern in intent['patterns']:
		        # tokenize each word in the sentence
		        w = nltk.word_tokenize(pattern)
		        #print(w)
		        # add to our words list
		        words.extend(w)
		        # add to documents in our corpus
		        documents.append((w, intent['tag']))
		        # add to our classes list
		        if intent['tag'] not in classes:
		            classes.append(intent['tag'])

		# stem and lower each word and remove duplicates
		words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
		words = sorted(list(set(words)))

		# remove duplicates
		classes = sorted(list(set(classes)))
		# classes = sorted(list(classes))

		# create our training data
		training = []
		output = []
		# create an empty array for our output
		output_empty = [0] * len(classes)

		# training set, bag of words for each sentence
		for doc in documents:
		    # initialize our bag of words
		    bag = []
		    # list of tokenized words for the pattern
		    pattern_words = doc[0]
		    # stem each word
		#     print(pattern_words)
		    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
		#     print("-------after stemming--------",pattern_words)
		    # create our bag of words array
		    for w in words:
		        bag.append(1) if w in pattern_words else bag.append(0)
		    #print("----------bag of words---------->",bag)
		    # output is a '0' for each tag and '1' for current tag
		    output_row = list(output_empty)
		    output_row[classes.index(doc[1])] = 1
		    #print(output_row)
		    training.append([bag, output_row])
		#print("TRAINING DATA---->",training)
		# shuffle our features and turn into np.array
		random.shuffle(training)
		training = np.array(training)

		# create train and test lists
		train_x = list(training[:,0])
		train_y = list(training[:,1])

		# reset underlying graph data
		tf.reset_default_graph()
		# Build neural network
		net = tflearn.input_data(shape=[None, len(train_x[0])])
		net = tflearn.fully_connected(net, 8)
		#net = tflearn.fully_connected(net, 8)
		net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
		net = tflearn.regression(net)

		# Define model and setup tensorboard
		model = tflearn.DNN(net, tensorboard_dir='./model_generic/tflearn_logs')
		# Start training (apply gradient descent algorithm)
		model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
		model.save('./model_generic/model.tflearn')
		# print (intents['intents'])
		# print("------------------------",filename)
		# save all of our data structures
		pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "./model_generic/training_data", "wb" ) )

		# Training doctor model
		# train_doctor_model()
	return "Success"

def res_model():	#loads the model and it's data structures. To be called when chat is started.
	# restore all of our data structures
	data = pickle.load( open( "./model_generic/training_data", "rb" ) )
	words = data['words']
	classes = data['classes']
	train_x = data['train_x']
	train_y = data['train_y']

	# import our chat-bot intents file
	filename = 'inputFiles/reception.json'
	with open(filename) as json_data:
	    intents = json.load(json_data)
	# print (intents)
	# load our saved model
	tf.reset_default_graph()
	net = tflearn.input_data(shape=[None, len(train_x[0])])
	net = tflearn.fully_connected(net, 8)
	#net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
	net = tflearn.regression(net)
	model = tflearn.DNN(net, tensorboard_dir='./model_generic/tflearn_logs')
	model.load('./model_generic/model.tflearn')
	return words,classes,intents,model

@sample.route('/response_from_chatbot',methods=['POST'])
def response_from_chatbot():
	# global words, classes, intents, model 
	# words, classes, intents, model = res_model()
	user_msg=request.get_data()
	print('@@@@@@@@@@@@@',user_msg)
	if(user_msg in ['clear context','forget me']):
		print('Clearing Context...')

		json_con={}
		json_con[userID]=''
		with open('context_info.json',mode='w') as json_data:
			json.dump(json_con,json_data)

		domain_module.initialise()
		chatbot_response = 'This chat is re-initialised.'
		# chatbot_response = 'Context has been cleared..'
	else:
		chatbot_response=response(user_msg)
		print('------',classify(user_msg),'!!!!!!!!!!!!!!!!')
		print('------',chatbot_response,'!!!!!!!!!!!!!!!!')
		if chatbot_response is None:
			chatbot_response = 'I did not get it.'
	return chatbot_response

# Changes Started: Sumitesh --19 Apr-2018 ####################
# xlToJsonConverter('excel.xls','data.json')
@sample.route("/xlToJsonConverter",methods=['POST'])
def xlToJsonConverter(inputExcel, outputJson):
    workbook = xlrd.open_workbook(inputExcel)
    worksheet = workbook.sheet_by_index(0)

    data = []
    keys = [v.value for v in worksheet.row(0)]
    for row_number in range(worksheet.nrows):
        if row_number == 0:
            continue
        row_data = {}
        for col_number, cell in enumerate(worksheet.row(row_number)):
            if cell.value=="":
                continue
            elif(col_number in [1,2]):
                e=cell.value.replace('", ','",').replace(' ,"',',"').split('",')
                e=[a.strip('"') for a in e]
                row_data[keys[col_number]] = e
            else:
                row_data[keys[col_number]] = cell.value

        data.append(row_data)
    with open(outputJson, 'w') as json_file:
        json_file.write(json.dumps({'intents': data}).replace('\"', '"'))


# Changes Ended: Sumitesh --19 Apr-2018 ####################

if __name__ == '__main__':
	sample.run()
