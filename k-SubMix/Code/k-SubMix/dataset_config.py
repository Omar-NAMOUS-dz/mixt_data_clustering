import os
cur_path = os.path.dirname(__file__)
data_path=os.path.join(cur_path, '..', '..','Datasets')
eval_path=os.path.join(cur_path, '..', '..','Evaluations')

scenarioEval='evaluation_test.csv'
exp1={
    "categorical_dims":"0,1,2",
    "numerical_dims":"3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19",
    "ground_dims":"20",
    "k_list":"3,1",
    "data_path":os.path.join(data_path,'exp1_processed.csv'),
    "evaluation_path" : os.path.join(eval_path, scenarioEval)
}

exp2={
    "categorical_dims":"3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19",
    "numerical_dims":"0,1,2",
    "ground_dims":"20",
    "k_list":"3,1",
    "data_path":os.path.join(data_path,'exp2_processed.csv'),
    "evaluation_path" : os.path.join(eval_path, scenarioEval)
}

S3={
    "numerical_dims":"0,1,2,3,4,5,6,7,8,9",
    "categorical_dims":"10,11,12,13,14,15,16,17,18,19",
    "ground_dims":"20",
    "k_list":"3,1",
    "data_path":os.path.join(data_path,'Scen2.csv'),
    "evaluation_path" : os.path.join(eval_path, scenarioEval)
}
