# traffic4cast-TLab

This repository is our solution to NeroIPS 2019 traffic4cast Competations.
For more information, please read this report (__to be updated__).
  
### How to run the codes.

1. run process.py

   e.g. `python process.py --city Moscow -i ./data/`
   
2. run train\_v17.py

   e.g. `python train_v17.py --city Moscow -ch 2 --crop 0 -o ./`
   
3. run test\_v17.py

   e.g. `python test_v17.py --city Moscow -ch 2 --crop 0 -o ./ -m ./`
   
4. run gen\_v17.py

   e.g. `python gen_v17.py --city Moscow `
