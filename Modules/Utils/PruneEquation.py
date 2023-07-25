import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def PruneEquation(theta_old_dict, y, alpha, max_pruning=-1) -> dict:
    '''
    Args:
        theta_old_dict (dict): dictionary containing the feature matrix that we wish to prune.
            'theta' -> feature matrix (np.ndarray)
            'features' -> 1d array of the feature names (default are indices).
        y (np.ndarray): target values.
        alpha (float): threshold value that determines pruning.
    Returns:
        theta_new_dict (dict): dictionary with key value pairs 
            'theta' -> pruned feature matrix
            'xi' -> coefficients from linear regression fit on pruned feature matrix and target values y.
            'pruned' -> boolean value indicating if pruning occured.
            'features' -> 1d array of the feature names.
            'old_features' -> 1d array of original feature names.
            'val' -> validation score measured in MSE.
            'val0' -> original validation score measured in MSE.
            #'change' -> percentage change in validation MSE between old and new. If pruned==False then 0%.
            'num_pruned' -> number of features pruned. If pruned==False then 0. If -1 then no max.
    '''
    theta_new_dict = dict()
    theta_new_dict['old_features'] = theta_old_dict['features']
    old_feature_arr = theta_new_dict['features']
    pruned_features = np.array([])
    
    theta_0 = theta_old_dict['theta']
    lm = LinearRegression()
    scoring = 'neg_mean_squared_error'
    cv0 = cross_val_score(lm, theta_0, y, cv=5, scoring=scoring)
    val0 = cv0.mean()*-1
    theta_new_dict['val0'] = val0
    
    def prune(lm, theta_curr, y, val_0, curr_feature_arr, alpha):
        '''
        Single pruning pass
        '''
        for i, name in enumerate(curr_feature_arr):
            theta_i = np.delete(theta_0, i, axis=1)
            cv_i = cross_val_score(lm, theta_i, y, cv=5, scoring='neg_mean_squared_error')
            val_i = cv_i.mean()*-1
            
            if val_i / val0 < 1 + alpha:
                pruned_features = np.append(pruned_features, name)
                return True, theta_i, val_i
            
            else:
                return False, theta_curr, val_0
    
    theta_curr = theta_0.copy()
    curr_feature_arr = old_feature_arr.copy()
    pruned = True
    num_pruned = 0
    while pruned:
        pruned, theta_curr, val_curr = prune(lm, theta_curr, y, val0, curr_feature_arr, alpha)
        if not pruned:
            num_pruned += 1
            
    theta_new_dict['theta'] = theta_curr
    theta_new_dict['xi'] = lm.coef_
    theta_new_dict['pruned'] = True if num_pruned > 0 else False
    theta_new_dict['features'] = np.array([])
    for elem in old_feature_arr:
        if elem not in pruned_features:
            theta_new_dict['features'] = np.append(theta_new_dict['features'], elem)
    theta_new_dict['val'] = val_curr
    theta_new_dict['num_pruned'] = num_pruned
    
    return theta_new_dict