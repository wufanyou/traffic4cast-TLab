# traffic4cast-TLab

This repository is our solution to NeroIPS 2019 traffic4cast Competations.

We rank 4th in the public leadboard.

For more information, please read `traffic4cast-report.pdf` in this repository.
  
### How to run the codes.

1. run process.py

   e.g. `python process.py --city Moscow -i ./data/`
   
   process.py: generate all useful features. In order to increase IO performance, each feature is a npy file.
   
   
2. run train\_v17.py

   e.g. `python train_v17.py --city Moscow -ch 2 --crop 0 -o ./`
   
   train\_v17.py: train models.
   
3. run test\_v17.py

   e.g. `python test_v17.py --city Moscow -ch 2 --crop 0 -o ./ -m ./`
   
   test\_v17.py: output prediction.
   
4. run gen\_v17.py

   e.g. `python gen_v17.py --city Moscow `
   
   gen\_v17.py: generate submission.
