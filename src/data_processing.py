import pandas as pd

#  Laoding the data
def load_data(filepath):
    """ Loads a CSV file into a pandas DataFrame """
    try:
        df = pd.read_csv(filepath)
        print(f'Success! Data loaded. The dataset has {df.shape[0]} rows and {df.shape[1]} columns')
        return df
    except FileNotFoundError:
        print(f'Error: could not find the file at {filepath}. Check your path!')
        return None
#  Checking for missing values
def check_missing_values(df):
    """Prints the summary of missing values in the dataframe"""
    missing_count = df.isna().sum()
    missing_pct = df.isna().mean()*100

    # creating a table for missing values
    missing_df = pd.DataFrame({
        'count': missing_count,
        'percentage': missing_pct
    })

    # Filtering to only show the missing columns
    missing_df = missing_df[missing_df['count']>0].sort_values(by='percentage', ascending=False)
    print('\n--- Missing Values Summary ---')
    print(missing_df)
    return missing_df

# cleaning the mort_acc

def fill_mort_acc(df):
    """Fills missing mort_acc values based on the mean of total_acc groups."""
    print("Correlating total_acc with mort_acc to fill gaps...")
    
    # Calculate the average mort_acc for every distinct total_acc count
    total_acc_avg = df.groupby('total_acc')['mort_acc'].mean()
    
    def fill_func(total_acc, mort_acc):
        if pd.isnull(mort_acc):
            return total_acc_avg[total_acc]
        else:
            return mort_acc

    df['mort_acc'] = df.apply(lambda x: fill_func(x['total_acc'], x['mort_acc']), axis=1)
    return df

# cleaning the rest of the missing values

def final_clean(df):
    """The final sweep: Handles all remaining missing values and text conversions."""
    print("Finalizing data cleaning...")

    # 1. Drop unnecessary columns
    # 'title' is redundant; 'emp_title' is too noisy
    df = df.drop(['title', 'emp_title'], axis=1)

    # 2. Fill 'emp_length' with the most common value (the mode)
    # We first convert it to a string to be safe, then fill
    df['emp_length'] = df['emp_length'].fillna(df['emp_length'].mode()[0])

    # 3. Drop the remaining tiny rows (revol_util and pub_rec_bankruptcies)
    df = df.dropna(subset=['revol_util', 'pub_rec_bankruptcies'])

    # 4. Handle 'term' (Convert " 36 months" -> 36)
    df['term'] = df['term'].str.extract(r'(\d+)').astype(int)

    # 5. Dropping the issue_d column for data leakage
    df = df.drop('issue_d', axis=1)

    print(f"Final Data Shape: {df.shape}")
    return df

# preparing the data

def prepare_features(df):
    """Translate all string columns into machine-readable numbers"""
    print('Translating strings to numerics features')

    # 1. Drop the useless/redundant strings
    df = df.drop(['grade', 'address', 'earliest_cr_line'], axis=1, errors='ignore')

    # 2. Map the target variable to 1s and 0s
    if 'loan_status' in df.columns:
        df['loan_repaid'] = df['loan_status'].map({
            'Fully Paid': 1, 
            'Charged Off': 0
        })
        df = df.drop('loan_status', axis=1)

    # 3. Map employment length to integers
    emp_map = {'< 1 year':0, '1 year':1, '2 years':2, '3 years':3, '4 years':4, 
               '5 years':5, '6 years':6, '7 years':7, '8 years':8, '9 years':9, '10+ years':10}
    if df['emp_length'].dtype == 'object' or df['emp_length'].dtype == 'str':
        df['emp_length'] = df['emp_length'].map(emp_map).fillna(0)

    # 4. one-hot encoding the remaining categoricals
    cat_cols = ['sub_grade', 'home_ownership', 'verification_status', 'purpose', 'initial_list_status', 'application_type']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True).astype(int)

    print(f'Features prepared! New shape: {df.shape}')
    return df

