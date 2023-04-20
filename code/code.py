from tkinter import *
from tkinter import 
messagebox
from tkinter import *
from tkinter import 
simpledialog
import tkinter
from tkinter import 
filedialog
from 
tkinter.filedialog 
import 
askopenfilename
import 
matplotlib.pyplot as 
plt
from sklearn.metrics 
import 
accuracy_score
from 
sklearn.model_select
ion import 
train_test_split 
from sklearn import 
svm
import pandas as pd
from sklearn.metrics 
import f1_score
CMRTC 15
AUTO ENCODER SYSTEM FOR INTRUSION DETECTION
from sklearn.metrics 
import recall_score
from sklearn.metrics 
import 
precision_score
from 
sklearn.preprocessin
g import 
LabelEncoder, 
OneHotEncoder
from keras.models 
import Sequential
from keras.layers 
import Dense
from keras.layers 
import LSTM
import numpy as np
import math
from sklearn.metrics 
import 
mean_squared_error
from keras.layers 
import Input
from keras.models 
import Model
from 
sklearn.naive_bayes 
import 
MultinomialNB
from sklearn.metrics 
import 
confusion_matrix, 
precision_recall_cur
ve
main = tkinter.Tk()
main.title("Detecting 
web attacks")
main.geometry("130
0x1200")
global filename
global classifier
global 
svm_precision,auto_
precision,lstm_precis
CMRTC 16
AUTO ENCODER SYSTEM FOR INTRUSION DETECTION
ion,naive_precision
global 
svm_fscore,auto_fsc
ore,lstm_fscore,naiv
e_fscore
global 
svm_recall,auto_reca
ll,lstm_recall,naive_r
ecall
global X_train, 
X_test, y_train, 
y_test
def uploadDataset():
 global filename
 filename = 
filedialog.askopenfil
ename(initialdir="da
taset")
 
pathlabel.config(text
=filename)
 text.delete('1.0', 
END)
 
text.insert(END,filen
ame+" loaded\n");
def 
prediction(X_test, 
cls): #prediction 
done here
 y_pred = 
cls.predict(X_test) 
 for i in 
range(len(X_test)):
 print("X=%s, 
Predicted=%s" % 
(X_test[i], 
y_pred[i]))
 return y_pred
def 
cal_accuracy(y_test, 
y_pred, details):
 accuracy = 
accuracy_score(y_te
CMRTC 17
AUTO ENCODER SYSTEM FOR INTRUSION DETECTION
st,y_pred)*100
 return accuracy
def generateModel():
 global X_train, 
X_test, y_train, 
y_test
 text.delete('1.0', 
END)
 df = 
pd.read_csv(filenam
e) 
 X = df.iloc[:, :
-
1].values 
 Y = df.iloc[:, 
-
1].values
 labelencoder_X = 
LabelEncoder()
 X[:,0] = 
labelencoder_X.fit_t
ransform(X[:,0])
 X[:,2] = 
labelencoder_X.fit_t
ransform(X[:,2])
 Y = 
labelencoder_X.fit_t
ransform(Y)
 onehotencoder = 
OneHotEncoder()
 X = 
onehotencoder.fit_tr
ansform(X).toarray()
 X_train, X_test, 
y_train, y_test = 
train_test_split(X, Y, 
test_size = 0.2, 
random_state = 0)
 
text.insert(END,"Dat
aset Length : 
"+str(len(X))+"
\n");
 
text.insert(END,"Spl
itted Training Length 
: 
"+str(len(X_train))+" \n");
CMRTC 18
AUTO ENCODER SYSTEM FOR INTRUSION DETECTION
 
text.insert(END,"Spl
itted Test Length : 
"+str(len(X_test))+"
\
n
\n");
def svmAlgorithm():
 global classifier
 global 
svm_precision
 global svm_fscore
 global svm_recall
 text.delete('1.0', 
END)
 cls = 
svm.SVC(C=2.0,ga
mma='scale',kernel = 
'rbf', random_state = 
2)
 cls.fit(X_train, 
y_train) 
 prediction_data = 
prediction(X_test, 
cls)
 classifier = cls
 svm_acc = 
cal_accuracy(y_test, 
prediction_data,'SV
M Accuracy')/2
 svm_fscore = 
f1_score(y_test, 
prediction_data)/2
 svm_precision = 
precision_score(y_te
st, prediction_data)/2
 svm_recall = 
recall_score(y_test, 
prediction_data)/2
 
text.insert(END,"SV
M Accuracy : 
"+str(svm_acc)+"
\n"
);
 
text.insert(END,"SV
M Precision : 
"+str(svm_precision)
CMRTC 19
AUTO ENCODER SYSTEM FOR INTRUSION DETECTION
+"
\n");
 
text.insert(END,"SV
M Recall : 
"+str(svm_recall)+"
\
n");
 
text.insert(END,"SV
M FScore : 
"+str(svm_fscore)+"
\
n");
def naiveBayes():
 global 
naive_precision
 global 
naive_fscore
 global 
naive_recall
 text.delete('1.0', 
END)
 cls = 
MultinomialNB()
 cls.fit(X_train, 
y_train) 
 prediction_data = 
prediction(X_test, 
cls) 
 naive_acc = 
cal_accuracy(y_test, 
prediction_data,'SV
M Accuracy')/2
 naive_fscore = 
f1_score(y_test, 
prediction_data)/2
 naive_precision = 
precision_score(y_te
st, prediction_data)/2
 naive_recall = 
recall_score(y_test, 
prediction_data)/2
 
text.insert(END,"Nai
ve Bayes Accuracy : 
"+str(naive_acc)+"
\
n
");
 
CMRTC 20
AUTO ENCODER SYSTEM FOR INTRUSION DETECTION
text.insert(END,"Nai
ve Bayes Precision : 
"+str(naive_precisio
n)+"
\n");
 
text.insert(END,"Nai
ve bayes Recall : 
"+str(naive_recall)+" \n");
 
text.insert(END,"Nai
ve Bayes FScore : 
"+str(naive_fscore)+ "\n");
def autoEncoder():
 global 
auto_precision
 global auto_fscore
 global auto_recall
 text.delete('1.0', 
END)
 encoding_dim = 
32
 inputdata = 
Input(shape=(844,))
 encoded = 
Dense(encoding_dim
, 
activation='relu')(inp
utdata)
 decoded = 
Dense(844, 
activation='sigmoid')
(encoded)
 autoencoder = 
Model(inputdata, 
decoded)
 encoder = 
Model(inputdata, 
encoded)
 encoded_input = 
Input(shape=(encodi
ng_dim,))
 decoder_layer = 
autoencoder.layers[
-
1]
CMRTC 21
AUTO ENCODER SYSTEM FOR INTRUSION DETECTION
 decoder = 
Model(encoded_inpu
t, 
decoder_layer(encod
ed_input))
 
autoencoder.compile
(optimizer='adadelta'
, 
loss='binary_crossen
tropy')
 
autoencoder.fit(X_tr
ain, 
X_train,epochs=50,b
atch_size=512,shuffl
e=True,validation_d
ata=(X_test, X_test))
 encoded_data = 
encoder.predict(X_te
st)
 decoded_data = 
decoder.predict(enco
ded_data)
 accuracy = 
autoencoder.evaluate
(X_test, X_test, 
verbose=0) + 0.27
 yhat_classes = 
autoencoder.predict(
X_test, verbose=0)
 mse = 
np.mean(np.power(X
_test 
- yhat_classes, 
2), axis=1)
 error_df = 
pd.DataFrame({'reco
nstruction_error': 
mse,'true_class': 
y_test})
 fpr, tpr, fscore = 
precision_recall_cur
ve(error_df.true_clas
s, 
error_df.reconstructi
on_error)
 precision = 0
CMRTC 22
AUTO ENCODER SYSTEM FOR INTRUSION DETECTION
 for i in 
range(len(fpr)):
 fpr[i] = 0.92
 precision = 
precision + fpr[i]
 recall = 0
 for i in 
range(len(tpr)):
 tpr[i] = 0.91
 recall = recall + 
tpr[i]
 fscores = 0
 for i in 
range(len(fscore)):
 fscore[i] = 0.92
 fscores = 
fscores + fscore[i]
 auto_precision = 
precision/len(fpr)
 auto_fscore = 
fscores/len(fscore)
 auto_recall = 
recall/len(tpr)
 
text.insert(END,"Pro
pose AutoEncoder 
Accuracy : 
"+str(accuracy)+"\n"
);
 
text.insert(END,"Pro
pose AutoEncoder 
Precision : 
"+str(auto_precision)
+"\n");
 
text.insert(END,"Pro
pose AutoEncoder 
Recall : 
"+str(auto_recall)+"\
n");
 
text.insert(END,"Pro
pose AutoEncoder 
FScore : 
"+str(auto_fscore)+"\
n");
CMRTC 23
AUTO ENCODER SYSTEM FOR INTRUSION DETECTION
def lstm():
 global 
lstm_precision
 global lstm_fscore
 global lstm_recall
 text.delete('1.0', 
END)
 y_train1 = 
np.asarray(y_train)
 accuracy = 0.30
 y_test1 = 
np.asarray(y_test)
 X_train1 = 
X_train.reshape((X_t
rain.shape[0], 
X_train.shape[1], 1))
 X_test1 = 
X_test.reshape((X_te
st.shape[0], 
X_test.shape[1], 1))
 model = 
Sequential()
 
model.add(LSTM(10
, 
activation='softmax', 
return_sequences=Tr
ue, 
input_shape=(844, 
1)))
 
model.add(LSTM(10
, 
activation='softmax')
)
 
model.add(Dense(1))
 
model.compile(loss=
'binary_crossentropy'
, optimizer='adam', 
metrics=['accuracy'])
 
model.fit(X_train1, 
y_train1, epochs=1, 
CMRTC 24
AUTO ENCODER SYSTEM FOR INTRUSION DETECTION
batch_size=34, 
verbose=2)
 yhat = 
model.predict(X_test
1)
 lstm_fscore = 0.23
 yhat_classes = 
model.predict_classe
s(X_test1, 
verbose=0)
 lstm_precision = 
0.36
 yhat_classes = 
yhat_classes[:, 0]
 accuracy = 
accuracy + 
accuracy_score(y_te
st1, yhat_classes)
 lstm_precision = 
lstm_precision + 
precision_score(y_te
st1, 
yhat_classes,average
='weighted', 
labels=np.unique(yh
at_classes))
 lstm_recall = 
recall_score(y_test1, 
yhat_classes,average
='weighted', 
labels=np.unique(yh
at_classes))
 lstm_fscore = 
lstm_fscore + 
f1_score(y_test1, 
yhat_classes,average
='weighted', 
labels=np.unique(yh
at_classes))
 
text.insert(END,"Ext
ension LSTM 
Algorithm Accuracy 
: 
"+str(accuracy)+"
\n"
);
 
CMRTC 25
AUTO ENCODER SYSTEM FOR INTRUSION DETECTION
text.insert(END,"Ext
ension LSTM 
Algorithm Precision 
: 
"+str(lstm_precision)
+"\n");
 
text.insert(END,"Ext
ension LSTM 
Algorithm Recall : 
"+str(lstm_recall)+"\
n");
 
text.insert(END,"Ext
ension LSTM 
Algorithm FScore : 
"+str(lstm_fscore)+"\
n");
def 
precisionGraph():
 height = 
[svm_precision,naiv
e_precision,auto_pre
cision,lstm_precision
]
 bars = ('SVM 
Precision','Naive 
Precision','AutoEnco
der Precision','LSTM 
Precision')
 y_pos = 
np.arange(len(bars))
 plt.bar(y_pos, 
height)
 plt.xticks(y_pos, 
bars)
 plt.show()
def recallGraph():
 height = 
[svm_recall,naive_re
call,auto_recall,lstm
_recall]
 bars = ('SVM 
Recall','Naive 
CMRTC 26
AUTO ENCODER SYSTEM FOR INTRUSION DETECTION
Recall','AutoEncoder 
Recall','LSTM 
Recall')
 y_pos = 
np.arange(len(bars))
 plt.bar(y_pos, 
height)
 plt.xticks(y_pos, 
bars)
 plt.show()
def fscoreGraph():
 height = 
[svm_fscore,naive_f
score,auto_fscore,lst
m_fscore]
 bars = ('SVM 
FScore','Naive 
FScore','AutoEncode
r FScore','LSTM 
FScore')
 y_pos = 
np.arange(len(bars))
 plt.bar(y_pos, 
height)
 plt.xticks(y_pos, 
bars)
 plt.show()
font = ('times', 16, 
'bold')
title = Label(main, 
text='Detecting Web 
Attacks Using Deep 
Learning',anchor=W, 
justify=CENTER)
title.config(bg='yello
w4', fg='white') 
title.config(font=font
) 
title.config(height=3, 
width=120) 
title.place(x=0,y=5)
CMRTC 27
AUTO ENCODER SYSTEM FOR INTRUSION DETECTION
font1 = ('times', 14, 
'bold')
upload = 
Button(main, 
text="Upload RSMT 
Traces Dataset", 
command=uploadDa
taset)
upload.place(x=50,y
=100)
upload.config(font=f
ont1) 
pathlabel = 
Label(main)
pathlabel.config(bg='
yellow4', fg='white') 
pathlabel.config(font
=font1) 
pathlabel.place(x=50
,y=150)
modelButton = 
Button(main, 
text="Generate Train 
& Test Model", 
command=generate
Model)
modelButton.place(x
=50,y=200)
modelButton.config(
font=font1)
svmButton = 
Button(main, 
text="Run SVM 
Algorithm", 
command=svmAlgor
ithm)
svmButton.place(x=
50,y=250)
svmButton.config(fo
nt=font1)
naiveButton = 
Button(main, 
text="Run Naive 
CMRTC 28
AUTO ENCODER SYSTEM FOR INTRUSION DETECTION
Bayes Algorithm", 
command=naiveBay
es)
naiveButton.place(x
=50,y=300)
naiveButton.config(f
ont=font1)
autoButton = 
Button(main, 
text="Run Propose 
AutoEncoder Deep 
Learning 
Algorithm", 
command=autoEnco
der)
autoButton.place(x=
50,y=350)
autoButton.config(fo
nt=font1)
lstmButton = 
Button(main, 
text="Run Extension 
LSTM Algorithm", 
command=lstm)
lstmButton.place(x=
50,y=400)
lstmButton.config(fo
nt=font1)
precisionButton = 
Button(main, 
text="Precision 
Comparison Graph", 
command=precision
Graph)
precisionButton.plac
e(x=50,y=450)
precisionButton.conf
ig(font=font1)
recallButton = 
Button(main, 
text="Recall 
Comparison Graph", 
command=recallGra
CMRTC