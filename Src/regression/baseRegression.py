import numpy as np
import pandas as pd
import statsmodels.api as sm

class BaseRegression:
    """Base class for regression models."""

    def __init__(self, addIntercept=True):
        self.addIntercept = addIntercept
        self.model = None
        self.results = None
    
    def fit(self, X, y):
        """Fit the OLS regression model using Statsmodels.
        
        Inputs:
        1. X (DataFrame)            : Predictor variables.
        2. y (Series)               : Response variable.

        Returns:
        1. model                    : Fitted OLS regression model using Statsmodels.
        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Add a column of ones for the intercept
        if self.addIntercept:
            # has_constant='add' is used to automatically add a column of ones to the DataFrame
            X = sm.add_constant(X, has_constant='add', prepend=True)
        
        # Fit the model
        model = sm.OLS(y, X)
        self.results_ = model.fit()
        self.model = model
        self.results = self.results_
        return self.model, self.results
    
    def predict(self, X)-> pd.Series:
        """Make predictions using the fitted OLS regression model.
        
        Inputs:
        1. X (DataFrame)            : Predictor variables.

        Returns:
        1. predictions (Series)     : Predicted values.
        """

        if self.results is None:
            raise ValueError("Model is not fitted yet. Call fit() before predict().")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if self.addIntercept:
            if 'const' not in X.columns:
                X = sm.add_constant(X, has_constant='add', prepend=True)
        return self.results_.predict(X)
        
    def summary(self):
        """Get the summary of the fitted OLS regression model.

        Returns:
        1. summary (str)           : Summary of the fitted model.
        """
        
        if self.results is None:
            raise ValueError("Model is not fitted yet. Call fit() before summary().")
        return self.results_.summary()

    def fitStepwise(self, X, y, p_value_threshold=0.05):
        """Perform stepwise regression using p-values.
        
        Inputs:
        1. X (DataFrame)                : Predictor variables.
        2. y (Series)                   : Response variable.
        3. p_value_threshold (float)    : Threshold for the p-value.
        
        Returns:
        1. model                        : Fitted stepwise regression model.
        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Begin with the full model
        features = list(X.columns)
        step = 0
        self.step_history_ = []

        # Keep track of the dropped features, p-values, and the R-squared-adjusted and R-squared values
        dropped = True

        while dropped:
            dropped = False
            # Fit the model on the current features
            X_current = X[features].copy()

            if self.addIntercept:
                X_current = sm.add_constant(X_current, has_constant='add', prepend=True)
            
            model = sm.OLS(y, X_current).fit()

            # Store the relevant values
            r2 = model.rsquared
            adj_r2 = model.rsquared_adj
            self.step_history_.append({
                'step': step,
                'features': features.copy(),
                'r2': r2,
                'adj_r2': adj_r2,
                'p_values': model.pvalues.drop('const', errors='ignore').to_dict()
            })

            # Find the feature with the highest p-value
            p_values = model.pvalues.drop('const', errors='ignore')
            max_p_value = p_values.max()
            feature_with_max_p_value = p_values.idxmax()

            # If the max p-value is greater than the threshold, drop the feature
            if max_p_value > p_value_threshold:
                dropped = True
                features.remove(feature_with_max_p_value)
           
            step += 1
        
        self.selected_features_ = features
        self._refitFinalModel(X, y, features)

        return self


    def _refitFinalModel(self, X, y, features):
        """Refit the final model using the selected features.
        
        Inputs:
        1. X (DataFrame)                : Predictor variables.
        2. y (Series)                   : Response variable.
        3. features (list)              : Selected features.

        """ 

        X_final = X[features].copy()
        if self.addIntercept:
            X_final = sm.add_constant(X_final, has_constant='add', prepend=True)
        
        self.final_model_ = sm.OLS(y, X_final).fit()
        self.results_ = self.final_model_
        self.final_summary_ = self.results_.summary()

        
    def getStepHistory(self):
        return pd.DataFrame(self.step_history_)