# human-activity-recognition-with-temporal-regularization
Implementing the main idea of this paper http://ieeexplore.ieee.org/document/7456502/.  Sensor based human activity recognition with temporal regularization 

For simple demonstration, we used the UCI human activity recognition dataset: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones. Put the dataset into the './data/' directory. 

Compare the LSTM method here: https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition, we are able to achieve higher accuracy with roughly defined super parameters and model structure, and we expect even higher accuracy with fine tunned parameters and datasets with good temporal information.

Overfitting seems to appear after many epochs, one possible solution is to add l2 or l2 regularization.  


Using TensorFlow backend.
 [*] Loaded parameters success!!!
 0/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.908288061619

 1/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.943274438381

 2/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.933763563633

 3/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.932404875755

 4/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.937839686871

 5/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.934103250504

 6/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.942255437374

 7/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.942595124245

 8/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.937160313129

 9/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.942255437374

 10/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.939877688885

 11/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.936141312122

 12/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.940896749496

 13/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.951086938381

 14/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.936141312122

 15/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.947350561619

 16/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.948369562626

 17/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.933423936367

 18/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.930366873741

 19/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.940557062626

 20/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.912364125252

 21/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.934442937374

 22/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.933763563633

 23/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.938858687878

 24/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.934442937374

 25/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.940557062626

 26/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.945652186871

 27/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.934442937374

 28/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.934442937374

 29/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.932744562626

 30/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.933423936367

 31/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.938179373741

 32/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.933763563633

 33/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.931725561619

 34/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.932744562626

 35/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.931725561619

 36/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.933423936367

 37/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.920855998993

 38/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.932404875755

 39/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.932065188885

 40/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.933423936367

 41/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.931046187878

 42/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.932065188885

 43/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.933084249496

 44/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.918478250504

 45/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.925951063633

 46/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.911684811115

 47/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.933763563633

 48/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.929347813129

 49/100 epoch, 113/114 batch. train_acc:0.5 val_acc:0.774796187878

 50/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.950407624245

 51/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.933423936367

 52/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.942595124245

 53/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.918478250504

 54/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.944293498993

 55/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.947010874748

 56/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.938179373741

 57/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.948369562626

 58/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.940896749496

 59/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.947010874748

 60/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.937160313129

 61/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.932404875755

 62/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.937839686871

 63/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.787364125252

 64/100 epoch, 113/114 batch. train_acc:1.0 val_acc:0.920516312122
