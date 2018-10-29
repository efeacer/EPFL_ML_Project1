Group: THREE COMMA CLUB (Efe Acer, Murat Topak, Daniil Dmitriev)

Steps to obtain our best prediction result on the Kaggle platform:
	
	1) Place the raw data sets 'train.cvs' and 'test.csv'  in this (Scripts) folder.
	
	2) Execute the run program with the data processing option to create processed data sets.
	
	3) If you want use the hardcoded hyperparamaters, you can simply ignore the grid search
	option of the run program. However, you can also obtain the same results by including 
	this option (Keep in mind that this is an exhaustive search and takes quite a lot of time)
	
	So what you need to do in your computer is to:
		
		a) Open terminal and navigate to this (Scripts) folder
		
		b) to execute 'run.py' type:
			
			i) python run.py -pd -> to process and create data sets and train the model using hard 
			coded hyperparameters (intended use for testing)
			
			ii) python run.py -pd -gs -> to process and create data sets, tune the hyperparameters
			with a grid search with cross validation and then train the model (takes some time)
			
			Note: You can omit the '-pd' option after the processed data sets are created in  this
			(Scripts) folder. You should not omit this option in the very first execution, otherwise
			the program will not find the processed data sets and crash.
			
	After running the program a file called 'Submit_E_M_D_best' will be present in this (Scripts)
	folder, which is indeed our best submission.
	
	You can find implementations of the six mandatory algorithms in the 'implementations.py' file,
	helper functions for these algorithms in the 'helper_functions.py' file and the functions required
	for data pre-processing and parameter tuning in the 'data_processing.py' file. All are located in 
	the Scripts folder together with 'proj1_helpers.py' that was already provided in the project skeleton.
	You will also see a file called 'test_implementations.py' that is used to test the six mandatory 
	algorithms and provide data for our report.
