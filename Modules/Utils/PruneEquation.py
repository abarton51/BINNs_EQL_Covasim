import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def PruneEquation(theta_old_dict, y, alpha, max_pruning=-1) -> dict:
    '''
    Args:
        theta_old_dict (dict): dictionary containing the feature matrix that we wish to prune.
            'theta' -> feature matrix (np.ndarray)
            'features' -> list of the feature names (default are indices).
        y (np.ndarray): target values.
        alpha (float): threshold value that determines pruning.
    Returns:
        theta_new_dict (dict): dictionary with key value pairs 
            'theta' -> pruned feature matrix
            'xi' -> coefficients from linear regression fit on pruned feature matrix and target values y.
            'pruned' -> boolean value indicating if pruning occured.
            'features' -> list of the feature names.
            'old_features' -> list of original feature names.
            'val' -> validation score measured in MSE.
            'val0' -> original validation score measured in MSE.
            #'change' -> percentage change in validation MSE between old and new. If pruned==False then 0%.
            'num_pruned' -> number of features pruned. If pruned==False then 0. If -1 then no max.
    '''
    theta_new_dict = dict()
    theta_new_dict['old_features'] = theta_old_dict['features']
    old_feature_arr = theta_new_dict['old_features']
    pruned_features = []
    
    theta_0 = theta_old_dict['theta']
    lm = LinearRegression()
    lm.fit(theta_0, y)
    theta_new_dict['xi'] = lm.coef_.T
    scoring = 'neg_mean_squared_error'
    cv0 = cross_val_score(lm, theta_0, y, cv=5, scoring=scoring)
    val0 = cv0.mean()*-1
    theta_new_dict['val0'] = val0
    
    def prune(lm, theta_curr, y, val_0, curr_feature_arr, pruned_features, alpha):
        '''
        Single pruning pass
        '''
        for i, name in enumerate(curr_feature_arr):
            theta_i = np.delete(theta_0, i, axis=1)
            lm.fit(theta_i, y)
            cv_i = cross_val_score(lm, theta_i, y, cv=5, scoring='neg_mean_squared_error')
            val_i = cv_i.mean()*-1
            
            if val_i / val0 < 1 + alpha:
                pruned_features.append(name)
                theta_new_dict['xi'] = lm.coef_.T
                return True, theta_i, val_i, pruned_features
            
            else:
                return False, theta_curr, val_0, pruned_features
    
    theta_curr = theta_0.copy()
    curr_feature_arr = old_feature_arr.copy()
    pruned = True
    num_pruned = 0
    if max_pruning == -1:   max_pruning = theta_0.shape[1]
    while pruned and num_pruned <= max_pruning:
        pruned, theta_curr, val_curr, pruned_features = prune(lm, theta_curr, y, val0, curr_feature_arr, pruned_features, alpha)
        for elem in curr_feature_arr:
            if elem in pruned_features:
                curr_feature_arr.remove(elem)
        if pruned:
            num_pruned += 1
            
    theta_new_dict['theta'] = theta_curr
    theta_new_dict['pruned'] = True if num_pruned > 0 else False
    theta_new_dict['features'] = []
    for elem in old_feature_arr:
        if elem not in pruned_features:
            theta_new_dict['features'].append(elem)
    theta_new_dict['val'] = val_curr
    theta_new_dict['num_pruned'] = num_pruned
    
    return theta_new_dict