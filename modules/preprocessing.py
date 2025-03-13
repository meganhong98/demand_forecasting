import pandas as pd
import os

def weekly_aggregation(transactions):
    """
    Aggregates transaction data by week in place and adds a new column for weekly aggregated transactions in thousands.

    Parameters:
        transactions (DataFrame): A DataFrame containing transaction data with a 't_dat' column.
    """
    # Ensure 't_dat' is in datetime format
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])

    # Aggregate transactions by week
    weekly_data = transactions.resample('W', on='t_dat').size()

    # Add a new column to the original DataFrame
    transactions['week'] = transactions['t_dat'].dt.to_period('W').dt.to_timestamp()
    weekly_data_map = weekly_data.to_dict()
    transactions['weekly_transactions'] = transactions['week'].map(weekly_data_map)

    return transactions 


def handle_missing_values(customers):
    customers['FN'] = customers['FN'].fillna(0)
    customers['Active'] = customers['Active'].fillna(0)
    customers['fashion_news_frequency'] = customers['fashion_news_frequency'].fillna('None')
    customers['age'] = customers['age'].fillna(customers['age'].median())
    customers['club_member_status'] = customers.apply(
        lambda row: 'ACTIVE' if pd.isna(row['club_member_status']) and row['fashion_news_frequency'] == 'Regularly'
        else ('NON-ACTIVE' if pd.isna(row['club_member_status']) else row['club_member_status']),
        axis=1
    )

    return customers

def group_articles(articles):
    # group articles together to reduce dimension size of dataset 
    grouped_articles = (
    articles.groupby(['product_type_name', 'colour_group_name', 'graphical_appearance_name'])
        .size() 
        .reset_index(name='article_count')  
    )
    articles = articles.merge(
        grouped_articles,
        on=['product_type_name', 'colour_group_name', 'graphical_appearance_name'],
        how='left'
    )
    return articles

def create_age_bins(customers):
    # Adjust age for specific groups
    customers['adjusted_age'] = customers['age']
    customers.loc[customers['age'] < 20, 'adjusted_age'] = 10
    customers.loc[customers['age'] >= 60, 'adjusted_age'] = 60

    # Define bins and labels for remaining ages
    age_bins = [10, 19, 29, 39, 49, 59, 69]
    age_labels = ['10-19', '20-29', '30-39', '40-49', '50-59', '60+']

    # Create an 'age_bin' column
    customers['age_bin'] = pd.cut(customers['adjusted_age'], bins=age_bins, labels=age_labels, right=False)

    # Fill missing values or categorize specific adjustments as '10-19' or '60+'
    customers.loc[customers['adjusted_age'] == 10, 'age_bin'] = '10-19'
    customers.loc[customers['adjusted_age'] == 60, 'age_bin'] = '60+'

    return customers


def purchase_rate_per_article_per_age(transactions, customers):
    """
    Computes the purchase rate of an article for each age bin and integrates it into the transactions dataset
    without adding the age_bin column directly.

    Parameters:
        transactions (DataFrame): A DataFrame containing transaction data with customer and article IDs.
        customers (DataFrame): A DataFrame containing customer data with 'customer_id' and 'age_bin'.

    Returns:
        DataFrame: The transactions DataFrame with a new 'purchase_rate' column.
    """
    # Merge age_bin from customers into transactions temporarily for calculation purposes
    temp_data = transactions.merge(customers[['customer_id', 'age_bin']], on='customer_id', how='left')

    # Calculate purchase rate for each article-age_bin combination
    purchase_rate = (
        temp_data.groupby(['article_id', 'age_bin'])
        .size()
        .div(temp_data.groupby('article_id').size(), level=0)  # Normalize by total article purchases
        .reset_index(name='purchase_rate')
    )

    # Aggregate purchase rate across all age bins for each article
    purchase_rate_summary = (
        purchase_rate.groupby('article_id')['purchase_rate']
        .mean()  # Adjust this if you want a different aggregation (e.g., max, sum)
        .reset_index()
    )

    # Merge the aggregated purchase rate back into the transactions dataset
    transactions = transactions.merge(purchase_rate_summary, on='article_id', how='left')
    
    return transactions


def calculate_elapsed_days(data, reference_date=None):
    """
    Calculates:
    1. Number of elapsed days from the last purchase among all customers.
    2. Number of elapsed days from the article's first sell.
    
    Parameters:
    - data: DataFrame containing transactions (with columns: 't_dat', 'customer_id', 'article_id').
    - reference_date: Optional; Use a specific date (e.g., today's date) for calculating elapsed days.
    
    Returns:
    - DataFrame with two new columns: 'elapsed_days_since_last_purchase' and 'elapsed_days_since_first_sell'.
    """
    # Ensure 't_dat' is in datetime format
    data['t_dat'] = pd.to_datetime(data['t_dat'])
    
    # Use today's date if no reference date is provided
    if reference_date is None:
        reference_date = pd.Timestamp.today()

    # Calculate the elapsed days since the last purchase for all customers
    last_purchase = data.groupby('customer_id')['t_dat'].max().reset_index()
    last_purchase['elapsed_days_since_last_purchase'] = (reference_date - last_purchase['t_dat']).dt.days

    # Merge the elapsed days back to the original dataset
    data = data.merge(last_purchase[['customer_id', 'elapsed_days_since_last_purchase']], on='customer_id', how='left')
    
    # Calculate the elapsed days since the article's first sell
    first_sell = data.groupby('article_id')['t_dat'].min().reset_index()
    first_sell['elapsed_days_since_first_sell'] = (reference_date - first_sell['t_dat']).dt.days

    # Merge the elapsed days back to the original dataset
    data = data.merge(first_sell[['article_id', 'elapsed_days_since_first_sell']], on='article_id', how='left')
    
    return data
    

def add_image_path_to_articles(articles, image_base_path="data/images/"):
    """
    Add an image_path column to the articles DataFrame if a corresponding image exists in the directory.

    Parameters:
    - articles: DataFrame containing article data (with 'article_id' column).
    - image_base_path: Base directory path where images are stored.

    Returns:
    - Updated DataFrame with a new 'image_path' column.
    """
    # Function to generate the image path
    def get_image_path(article_id):
        # Extract the first three digits of the article_id
        folder = str(article_id)[:3]
        # Construct the image file name
        file_name = f"{str(article_id)}.jpg"
        # Construct the full path
        full_path = os.path.join(image_base_path, folder, file_name)
        # Check if the file exists
        if os.path.exists(full_path):
            return full_path
        else:
            return None

    # Apply the function to the article_id column
    articles['image_path'] = articles['article_id'].apply(get_image_path)
    
    return articles
    
if __name__ == "__main__":
    # Load datasets
    transactions = pd.read_csv("data/transactions.csv")
    customers = pd.read_csv("data/customers.csv")
    articles = pd.read_csv("data/articles.csv")

    print("Starting preprocessing")

    print("Aggregating transactions weekly")
    transactions = weekly_aggregation(transactions)

    print("Grouping articles")
    articles = group_articles(articles)

    print("Handling missing values in customers dataset")
    customers = handle_missing_values(customers)

    print("Creating age bins for customers")
    customers = create_age_bins(customers)

    print("Calculating purchase rates per article per age bin")
    transactions = purchase_rate_per_article_per_age(transactions, customers)

    print("Calculating elapsed days from the last purchase and first sell")
    transactions = calculate_elapsed_days(transactions)

    print("Adding image paths to the articles dataset")
    articles = add_image_path_to_articles(articles, image_base_path="data/images/")

    print("Preprocessing complete!")

    # Save the processed datasets
    transactions.to_csv("data/processed/transactions.csv", index=False)
    customers.to_csv("data/processed/customers.csv", index=False)
    articles.to_csv("data/processed/articles.csv", index=False)