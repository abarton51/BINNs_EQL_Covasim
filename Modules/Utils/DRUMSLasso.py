import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


def DRUMS_Lasso(
    input_dict      : dict[str, np.ndarray],
    lhs_values      : np.ndarray,
    degree          : int = 2,
    alphas          : np.ndarray = np.array([0.0]),
    intercept       : bool = True,
    cv              : int = 5,
) -> dict:
    
    ''' 
        Purpose --> Take in left-hand side (LHS) values, and a dictionary of right-hand side (RHS)
        values. Then, do some combination of RHS terms to minimize the difference between the
        combinated RHS and the LHS values.
        
        Return --> A dictionary of (1) "Lasso" : the LassoCV object, (2) "Equation" : the string
        equation of the best fitting curve, (3) "MSE" : the mean squared error between the predicted
        values and the true LHS.
        
        Parameters:
            input_dict {Dictionary} : The keys are the string, variable symbol or terms used in the
            RHS, and the values are the ndarray of the possible values in the domain.
            
            lhs_values {numpy.ndarray} : From the domain, we have the corresponding range of values of
            the surface which we are trying to fit.
            
            degree {int} : The max power of the RHS. Default is degree = 2, so terms like X^2 and XY
            will be the maximum power terms.
            
            alphas {numpy.ndarray} : Can be specified for multiple values, but default is just one value.
            One can specify how many terms the fitted equation will accept. A lower alpha will
            correspond to more terms. alphas = [0, 1]
            
            intercept {bool} : Whether or not the resulting equation should solve for an intercept. The
            default is intercept = True
            
            cv {int} : The cross validation splitting strategy. Default, cv = 5.
    '''

    # Set up the input terms, data_x for input into Lasso
    input_terms = []
    data_x = None
    for var_name, np_vals in input_dict.items():
        input_terms.append(var_name)
        if data_x is None:
            data_x = np_vals[:, None]
        else:
            data_x = np.hstack([data_x, np_vals[:, None]])
    
    # Do all the combinations up to degree of the input variables
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(data_x)
    
    # Create a Lasso object and fit the data
    lasso = LassoCV(fit_intercept=intercept, cv=cv, alphas=alphas)
    lasso.fit(X_poly, lhs_values)

    # Get the coefficients and feature names
    coefs = lasso.coef_
    intercept = lasso.intercept_
    feature_names = poly.get_feature_names_out(input_features=input_terms)

    # Print the equation
    equation = "f = "
    for coef, name in zip(coefs, feature_names):
        if abs(coef) > 1e-6:
            equation += f"{coef:.5f}*{name} + "
    equation = equation[:-3] + f' + {intercept:.5f}'
    
    # Make predictions on the training data
    y_pred = lasso.predict(X_poly)

    # Calculate the training MSE
    mse = mean_squared_error(lhs_values, y_pred)
    
    output_dict = {
        'Lasso' : lasso,
        'Equation' : equation,
        'MSE' : mse,
        'degree' : degree
    }
    
    return output_dict


'''

# This is an example of DRUMS_LASSO


def arbitrary_func2(X : np.ndarray, Y : np.ndarray) -> np.ndarray:
    F = 0.45 * X**2 - 9.2 * X * Y
    return F

X = np.linspace(4, 5, 50)
Y = np.linspace(-1, 1, 50)
XX, YY = np.meshgrid(X, Y)
F = arbitrary_func2(XX, YY)

input_dict = {'X' : XX.ravel(), 'Y' : YY.ravel()}
lhs_values = F.ravel()

lasso = DRUMS_LASSO(input_dict, lhs_values)

print(lasso['Equation'])
print(lasso['MSE'])

'''