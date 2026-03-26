# GEO5017 
Assignment 2 
**Group 6:**

Arda Baysal  (5484987)
Emil Houb   (6550940)
Moritz Cermann (6562906)
Marta Żołnowska (6583725)
- - - 

**Running instructions:**
The code classifies urban objects (buildings, cars, fences, poles, trees) from 3D point cloud data (.xyz files) with two machine learning classifiers: Support Vector Machine (SVM) and Random Forest (RF).

Update the data path at the bottom of the script before running:
path = 'Data/pointclouds'  

Run python main.py 
Control which analysis steps are executed via the flags near the bottom of
the script:
 
    run_featureselector        = False  # Sequential forward feature selection
    run_hyperparameter_tuning  = False  # Grid search for best SVM / RF params
    run_training_CV            = True   # 5-fold cross-validation on training set
    run_learning_curve         = False  # Learning curves + confusion matrices
    run_SVM_classification     = True   # Final SVM evaluation on test set
    run_RF_classification      = True   # Final RF evaluation on test set

The default configuration (as shown above) runs cross-validation and final
classification for both SVM and RF using the pre-selected feature subset.
Setting the run_featureselector and run_hyperparameter_tuning to true makes the running time extensively long, thus for quick insight into the code and results one should set them to False. 

Output files: data.txt (feature file for all input poin cloud data), learning_curve_SVM.png, learning_curve_RF.png, confusion_matrix_SVM.png, confusion_matrix_RF.png.
Console output (J-scores, CV scores, accuracy) is printed to stdout only.
 
