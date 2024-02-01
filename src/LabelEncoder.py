from sklearn.preprocessing import LabelEncoder

def label_encode(df):
    
    le = LabelEncoder()
    # Drop sex duplicates
    le.fit(df['sex'].drop_duplicates()) 
    df['sex'] = le.transform(df['sex'])
    # Smoker or not
    le.fit(df['smoker'].drop_duplicates()) 
    df['smoker'] = le.transform(df['smoker'])
    # Region
    le.fit(df['region'].drop_duplicates()) 
    df['region'] = le.transform(df['region'])
    return df


def feature_drop(df):
    X = df.drop(['region'], axis = 1)
    return X

def polynomial_split(df):
    quad = PolynomialFeatures(degree = 3)
    x_quad = quad.fit_transform(df)

    return x_quad