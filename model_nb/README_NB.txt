Le script model_nb.py

Ce script a pour l'objectif d‘entraîner un model de classification sur les commentaires chinois à partir de la méthode multinomial NB et la représentation de Bow. Et il peut également lire un model déjà enregistré et prédire la catégorie des phrases. 
	1. OS : Windows, Mac, Linux
	2. Version du python : python3 ou plus 
	3. Librairies nécessaires: sklearn, pandas, nltk, matplotlib,joblib, jieba
	4. Les fichiers de train et test sont train_fileD.csv, test_fileD.csv
	5. Fichier(s) de model : NB_model.m , countvectorizer_model.m . Ils doivent être dans le même dossier du script.
	6. Usage : 
	train_model = ModelNB()
	Cela permet de créer une instance de la classe ModelNB().

	(1) Pour entraîner un model, vous allez dans le script et au fond, et utilisez/décommenter le code au-dessous : 
    	train_model.pipeline(save_model=False)

	'sava_model = False' veut dire de ne pas enregistrer le model dans vos ordinateur. 
	
	(2) Pour load a model enregistré, veuillez utiliser/décommenter le code au-dessous :
		train_model.load_model('NB_model.m','countvectorizer_model.m')
    
    (3) Pour prédire la polarité des phrases, veuillez utiliser/décommenter les codes au-dessous : 
		test_sentence = ["蒙牛好喝 ",'屏幕太小了','信号很差','手机还可以','手写识别还可以','苹果手机太贵']
    		predict_label_list = train_model.predict_sentence(test_sentence)
    		print(f'Prediction result : {predict_label_list}')

   	7. Si vous voulez modifier/changer les fichiers et train,  veuillez allez dans : 
   		 def __init__(self):
        	self.train_file = "train_fileD.csv"
        	self.test_file = "test_fileD.csv"

       De même, si vous voulez changer les fichiers de model, veuillez allez dans : 
       	def load_model(self,model_file,vector_file):
       		 print(f"Loading model {model_file}...")
        self.vectorizer = joblib.load(vector_file)
        print(f"Loading model {vector_file}...")
        self.naive= joblib.load(model_file) 


	