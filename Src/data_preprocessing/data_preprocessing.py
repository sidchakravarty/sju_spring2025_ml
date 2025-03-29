import numpy as np
import pandas as pd
from datetime import datetime, date
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

class PreProcessData:
    """This class loads the data, cleans it, and prepares it for modeling.
    
    Specific tasks include:
    1. Load data from a CSV or excel file.
    2. Explore the dataset to check for missing values.
    3. Impute the data (separately for numerical and categorical variables).
    4. Normalize or standardize the numerical data using Min-Max scaling.
    5. Create dummy variables for categorical variables with user-chosen prefix.
    """

    def __init__(self, scaleType: str ='minmax'):
        """Initialize the PreProcessData class.
        
        Inputs:
        1. scaleType (str)      : The type of scaling to be applied to numerical data. 
           a. "Standardize"     : StandardScaler.
           b. "Normalize"       : MinMaxScaler.
        """
        self.scaleType = scaleType
        self.scaler = None # This will set based on the scale_type

        if self.scaleType == 'standardize':
            self.scaler = StandardScaler()
        elif self.scaleType == 'normalize':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scale_type must be either 'standardize' or 'normalize'.")
        
        self.data = None # While initializing the class, we don't have the data yet.
        self.numerical_imputer = None # This will be set in the impute_data method.
        self.categorical_imputer = None # This will be set in the impute_data method.
        self.outlier_method = None # This will be set in the detect_outliers method.
    
    def loadData(self, filePath: str, fileFormat: str='csv', **kwargs):
        """Load data from a CSV or Excel file.
        
        Inputs:
        1. filePath (str)       : Path to the file.
        2. fileFormat (str)     : Format of the file. Options are 'csv' or 'excel'.
        3. **kwargs             : Additional arguments for pd.read_csv or pd.read_excel.
        
        Returns:
        1. data (DataFrame)     : Loaded data as a pandas DataFrame.
        """
        
        if fileFormat == 'csv':
            self.data = pd.read_csv(filePath, **kwargs)
        elif fileFormat == 'excel':
            self.data = pd.read_excel(filePath, **kwargs)
        else:
            raise ValueError("fileFormat must be either 'csv' or 'excel'.")

        print(f"Data loaded successfully from {filePath}.")
    
    def exploreData(self, displayRows: int = 5):
        """Display basic information about the loaded data.
        
            Inputs:
            1. displayRows (int)    : Number of rows to display. Defaults to 5.

            1. First few rows of the data. Defaults to 5 rows.
            2. Summary info
            3. Missing value counts
        """

        if self.data is None:
            raise ValueError("No data loaded. Please load data using the loadData method.")
        print("-------------------- Data Overview --------------------")
        print("-------------------- Data Snapshot --------------------")
        print(self.data.head(displayRows))
        print("\n-------------------- Data Info --------------------")
        print(self.data.info())
        print("\n-------------------- Missing Values Counts --------------------")
        print(self.data.isnull().sum())
    
    def detectOutliers(self, columnName: str, method: str = 'zscore', threshold: float = 3):
        """Detect outliers in a numerical column using a chosen method.
        
        Inputs:
        1. columnName (str)     : Name of the numerical column.
        2. method (str)         : Method to detect outliers. Options are 'zscore' or 'iqr'.
        3. threshold (float)    : Threshold for outlier detection.
        """

        if method == 'zscore':
            meanValue = self.data[columnName].mean()
            stdValue = self.data[columnName].std()
            z_scores = (self.data[columnName] - meanValue) / stdValue
            outliers = self.data[np.abs(z_scores) > threshold]
            return outliers
        elif method == 'iqr':
            Q1 = self.data[columnName].quantile(0.25)
            Q3 = self.data[columnName].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (1.5 * IQR)
            upper_bound = Q3 + (1.5 * IQR)
            outliers = self.data[(self.data[columnName] < lower_bound) | (self.data[columnName] > upper_bound)]
            return outliers
        else:
            raise ValueError("method must be either 'zscore' or 'iqr'.")
    
    def imputeData(self,
                   numericalImputeStrategy: str = 'mean',
                   categoricalImputeStrategy: str = 'most_frequent',
                   outlierDetectionMethod: str = 'zscore',
                   treatOutliers: bool = False,
                   outlierThreshold: float = 3):
        """Impute missing values in the dataset.
        
        Inputs:
        1. numericalImputeStrategy (str)    : Strategy to impute missing values in numerical columns.
              a. 'mean' : Mean imputation.
              b. 'median' : Median imputation.
              c. 'most_frequent' : Most frequent value imputation.
              d. 'constant' : Constant value imputation.
        2. categoricalImputeStrategy (str)  : Strategy to impute missing values in categorical columns.
              a. 'most_frequent' : Most frequent value imputation.
              b. 'constant' : Constant value imputation.
        3. outlierDetectionMethod (str) : Method to detect outliers. Options are 'zscore' or 'iqr'.
        4. treatOutliers (bool)        : Whether to treat outliers.
        5. outlierThreshold (float) : Threshold for outlier detection.
        """

        if self.data is None:
            raise ValueError("No data loaded. Please load data using the loadData method.")
        
        # Separate numerical and categorical columns
        numerical_columns = self.data.select_dtypes(include=['number']).columns
        
        # Detect and treat outliers
        if treatOutliers:
            for column in numerical_columns:
                outliersMask = self.detectOutliers(column,
                                                   method=outlierDetectionMethod,
                                                   threshold=outlierThreshold)
                self.data.loc[outliersMask.index, column] = np.nan # Replace outliers with NaN and then impute them later on
            
        # Impute numerical data
        self.numerical_imputer = SimpleImputer(strategy=numericalImputeStrategy)
        self.data[numerical_columns] = self.numerical_imputer.fit_transform(self.data[numerical_columns])
        print("Numerical data imputed successfully using strategy = {numericalImputeStrategy}.")

        # Impute categorical data
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        if len(categorical_columns) == 0:
            print("No categorical columns found to impute.")
            return
        else:
            self.categorical_imputer = SimpleImputer(strategy=categoricalImputeStrategy)
            self.data[categorical_columns] = self.categorical_imputer.fit_transform(self.data[categorical_columns])
            print("Categorical data imputed successfully using strategy = {categoricalImputeStrategy}.")

    
    def scaleData(self):
        """Scale numerical data using the specified scaler.
        
        Inputs:
        1. scaleType (str)      : The type of scaling to be applied to numerical data. 
           a. "Standardize"     : StandardScaler.
           b. "Normalize"       : MinMaxScaler.
        """
        
        if self.data is None:
            raise ValueError("No data loaded. Please load data using the loadData method.")
        
        # Separate numerical columns
        numerical_columns = self.data.select_dtypes(include=['number']).columns
        
        # Scale numerical data
        self.data[numerical_columns] = self.scaler.fit_transform(self.data[numerical_columns])
        print("Data scaled successfully using {self.scaleType} method.")
    
    def encodeCategoricalData(self, prefix:str = "cat"):
        """Create dummy variables for categorical variables with a user-chosen prefix.
        
        Inputs:
        1. prefix (str)        : Prefix for the dummy variable columns.
        
        """

        if self.data is None:
            raise ValueError("No data loaded. Please load data using the loadData method.")
        
        # Separate categorical columns
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        if len(categorical_columns) == 0:
            print("No categorical columns found to encode.")
            return
        else:
            # Create dummy variables
            self.data = pd.get_dummies(self.data, prefix=prefix, columns=categorical_columns, drop_first=True)
            print("Categorical data encoded successfully using prefix '{prefix}'.")
    
    def saveData(self, filePath: str, fileFormat: str='csv'):
        """Save the processed data to a CSV or Excel file.
        
        Inputs:
        1. filePath (str)       : Path to the file.
        2. fileFormat (str)     : Format of the file. Options are 'csv' or 'excel'.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data using the loadData method.")
        
        if fileFormat == 'csv':
            self.data.to_csv(filePath, index=False)
        elif fileFormat == 'excel':
            self.data.to_excel(filePath, index=False)
        else:
            raise ValueError("fileFormat must be either 'csv' or 'excel'.")
        
        print(f"Data saved successfully to {filePath}.")

    def getData(self):
        """Return the processed data."""
        return self.data
