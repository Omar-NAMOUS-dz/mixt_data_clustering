from dataset_config import *
import numpy as np

if __name__ == "__main__":
    """_________________USER INPUT____________________"""
    """SET DATASET, GAMMALIST AND NUMBER OF ITERATIONS"""
    """_______________________________________________"""
    dataset = exp1  #Choose from one of the datasets as defined in dataset_config.py
    gammaList=np.arange(0.1,1,0.01).tolist() #Choose gamma parameter regulating trade-off between numerical and categorical costs, define range and step size
    number_of_iterations_kSubMix = 10 #Number of k-SubMix iterations
    """____________________________________________________"""
    """________________END USER INPUT______________________"""

    numerical_dimensions = dataset["numerical_dims"]
    categorical_dimensions = dataset["categorical_dims"]
    ground_truth = dataset["ground_dims"]
    k_list = dataset["k_list"]
    data_path = dataset["data_path"]
    evaluation_path = dataset["evaluation_path"]

for gamma in gammaList:
    print("Gamma= "+str(gamma))
    for i in range(number_of_iterations_kSubMix):
        error = True
        while error == True:
            try:
                print("Iteration "+str(i))
                os.system("python3 k_SubMix.py {0} {1} {2} {3} {4} {5} {6}".format(
                    k_list,numerical_dimensions,categorical_dimensions,ground_truth,gamma,data_path,evaluation_path))
                error = False
            except Exception as e:
                print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
                error = True
    print()