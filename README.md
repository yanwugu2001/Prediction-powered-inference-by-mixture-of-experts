# Prediction-Powered Inference by Mixture of Experts                                                                                                                      
                                                                                                                                                                            
  This is the code repository for the paper **"Prediction-Powered Inference by Mixture of Experts"**.                                                                       
                                                                                                                                                                            
  ## Overview                                                                                                                                                               
                                                                                                                                                                            
  This repository contains the implementation and experiments for prediction-powered inference (PPI) using a mixture-of-experts framework. The code supports both numerical 
  simulations and real-data experiments.                                                                                                                                    
                                                                                                                                                                            
  ## Repository Structure

    .                                                                                                                                                                         
    ├── experiment/
    │   ├── numerical_simulation.ipynb   # Numerical simulation experiments                                                                                                   
    │   ├── ppi_tools.py                 # Core PPI statistical methods                                                                                                     
    │   ├── plot_tools.py                # Basic plotting utilities
    │   └── plot_tools_professional.py  # Publication-quality plotting
    ├── real_data/
    │   └── real_data_experiments.ipynb  # Real-data experiments
    └── dataset/                                                                                                                                                              
        └── tabular/
            └── raw/                                                                                                                                                          
                ├── bike_sharing.csv                                                                                                                                        
                └── california_housing.csv

  ## Requirements
                                                                                                                                                                            
  numpy
  pandas                                                                                                                                                                    
  scipy                                                                                                                                                                   
  scikit-learn
  xgboost
  lightgbm
  torch
  matplotlib
  seaborn                                                                                                                                                                   
  tqdm
                                                                                                                                                                            
  Install dependencies via:                                                                                                                                               

  ```bash
  pip install numpy pandas scipy scikit-learn xgboost lightgbm torch matplotlib seaborn tqdm
                                                                                                                                                                            
  Usage
                                                                                                                                                                            
  Numerical Simulations                                                                                                                                                   

  Open and run experiment/numerical_simulation.ipynb. All methods are implemented in ppi_tools.py.                                                                          
   
  Real-Data Experiments                                                                                                                                                     
                                                                                                                                                                          
  Open and run real_data/real_data_experiments.ipynb. The notebook reads data from dataset/tabular/raw/ using relative paths — keep the directory structure intact.         
   
  Citation                                                                                                                                                                  
                                                                                                                                                                          
  If you find this code useful, please cite our paper:

  @article{xxx,                                                                                                                                                             
    title={Prediction-Powered Inference by Mixture of Experts},
    author={},                                                                                                                                                              
    journal={},                                                                                                                                                           
    year={2025}
  }
