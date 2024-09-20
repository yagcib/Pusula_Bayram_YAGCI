import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# Load the dataset
def load_data(file_path):
    return pd.read_excel(file_path)


file_path = 'side_effect_data 1.xlsx'
data = load_data(file_path)


# Exploratory Data Analysis
def check_missing_values(data):
    return data.isnull().sum()


missing_values = check_missing_values(data)


def visualize_missing_data(data):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Data Heatmap")
    plt.show()


visualize_missing_data(data)


# Convert Dogum_Tarihi to calculate age
def calculate_age(data):
    data['Dogum_Tarihi'] = pd.to_datetime(data['Dogum_Tarihi'], errors='coerce')
    data['Ilac_Baslangic_Tarihi'] = pd.to_datetime(data['Ilac_Baslangic_Tarihi'], errors='coerce')
    current_year = pd.to_datetime("today").year
    data['Age'] = current_year - data['Dogum_Tarihi'].dt.year
    return data


data = calculate_age(data)


# Plotting distribution of Age
def plot_age_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Age'].dropna(), bins=30, kde=True)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()


plot_age_distribution(data)


# Plotting distribution of Gender
def plot_gender_distribution(data):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Cinsiyet', data=data)
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.show()


plot_gender_distribution(data)


# Plotting distribution of side effects
def plot_side_effects_distribution(data):
    plt.figure(figsize=(12, 6))
    sns.countplot(y='Yan_Etki', data=data, order=data['Yan_Etki'].value_counts().index)
    plt.title('Side Effects Distribution')
    plt.ylabel('Side Effect')
    plt.xlabel('Count')
    plt.show()


plot_side_effects_distribution(data)


# Data Preprocessing
def handle_missing_values(data):
    # Impute numeric columns separately
    kilo_imputer = SimpleImputer(strategy='mean')
    data['Kilo'] = kilo_imputer.fit_transform(data[['Kilo']])

    boy_imputer = SimpleImputer(strategy='mean')
    data['Boy'] = boy_imputer.fit_transform(data[['Boy']])

    # Impute categorical columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    data['Kan Grubu'] = cat_imputer.fit_transform(
        data[['Kan Grubu']]).ravel()  # Tek boyutlu hale getirmek için .ravel() kullan

    return data


data = handle_missing_values(data)


def encode_categorical_variables(data):
    # sparse yerine sparse_output kullanıyoruz
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_columns = encoder.fit_transform(data[['Cinsiyet', 'Uyruk', 'Il', 'Yan_Etki']])
    encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out())
    return encoded_df

encoded_df = encode_categorical_variables(data)


def merge_encoded_columns(data, encoded_df):
    data_preprocessed = pd.concat([data, encoded_df], axis=1)
    data_preprocessed.drop(['Cinsiyet', 'Uyruk', 'Il', 'Yan_Etki'], axis=1, inplace=True)
    return data_preprocessed


data_preprocessed = merge_encoded_columns(data, encoded_df)