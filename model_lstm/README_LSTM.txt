Le script model_lstm.py

Ce script a pour l'objectif d‘entraîner un model de classification sur les commentaires chinois à partir de la méthode LSTM. Et il peut également lire un model déjà enregistré et prédire la catégorie des phrases. 
	1. OS : Windows, Mac, Linux
	2. Version du python : python3 ou plus 
	3. Librairies nécessaires: sklearn, pandas, nltk, matplotlib,joblib, jieba,keras,numpy
	4. Les fichiers utilisé est : corpus.csv
	5. Fichier(s) de model : lstm_model.h5 , lstm_tokenizer.m . Ils doivent être dans le même dossier du script.
	6. Usage : 
	lstm_model = Model_LSTM()
	Cela permet de créer une instance de la classe Model_LSTM().

	(1) Pour entraîner un model, vous allez dans le script et au fond, et utilisez/décommenter le code au-dessous : 
    	# lstm_model.pipeline(save_model=False)

	'sava_model = False' veut dire de ne pas enregistrer le model dans vos ordinateur. 
	
	(2) Pour load a model enregistré, veuillez utiliser/décommenter le code au-dessous :
		lstm_model.load_model('lstm_model.h5','lstm_tokenizer.m')
    
    (3) Pour prédire la polarité des phrases, veuillez utiliser/décommenter les codes au-dessous : 
		test_sentence = ["蒙牛好喝 ",'用了几次发现头好痒，感觉过敏了','信号很差','用了几天就好卡，上当了，退款','手写识别还可以','房间挺大，就是价格贵了点']
    		lstm_model.predict_sentence(test_sentence)

   	7. Si vous voulez modifier/changer le fichier veuillez allez dans : 
   		 self.df = pd.read_csv('corpus.csv')

       De même, si vous voulez changer les fichiers de model, veuillez allez dans : 
       	def load_model(self,model_file,vector_file):
       		 print(f"Loading model {model_file}...")
        self.lstm_model = load_model(model_file)
        print(f"Loading model {tokenizer_file}...")
        self.tokenizer = joblib.load(tokenizer_file)   


	