import re
import pandas as pd
from modules.preprocessing import *

"""
Fix file and folder names by replacing empty spaces with underscores 
"""
def process_name(filename):
    filename = filename.replace(" ", "_")
    return re.sub(r'[^\w\-_\.]', '_', filename)

import pandas as pd

def create_grouped_data(customers, articles, transactions):
    """
    Processes customers, articles, and transactions data to create a dataframe with grouped products
    with product_type_name, colour_group_name, and graphical_appearance_name, and stores the original keywords.

    Parameters:
        customers (pd.DataFrame): Customers DataFrame.
        articles (pd.DataFrame): Articles DataFrame.
        transactions (pd.DataFrame): Transactions DataFrame.

    Returns:
        pd.DataFrame: The processed and grouped DataFrame.
    """
    # Handle missing values and preprocess data
    customers = handle_missing_values(customers)
    customers = create_age_bins(customers)
    transactions = calculate_elapsed_days(transactions)

    # Merge datasets
    transactions_articles = pd.merge(transactions, articles, on='article_id', how='inner')
    data = pd.merge(transactions_articles, customers, on='customer_id', how='left')

    # Group product attributes
    data['product_group'] = (
        data['product_type_name'] + ' ' +
        data['colour_group_name'] + ' ' +
        data['graphical_appearance_name']
    )

    # Store the original keywords (product_type_name, colour_group_name, graphical_appearance_name)
    data['product_type_keyword'] = data['product_type_name']
    data['colour_group_keyword'] = data['colour_group_name']
    data['graphical_appearance_keyword'] = data['graphical_appearance_name']

    # Rename the column from 't_dat' to 'date'
    data.rename(columns={'t_dat': 'date'}, inplace=True)

    # Convert the 'date' column to datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Aggregate by date and product group, and compute required features
    grouped_data = data.groupby(['date', 'product_group']).agg(
        transaction_count=('article_id', 'count'),  # Total transaction count
        avg_price=('price', 'mean'),              # Average price
        sales_channel=('sales_channel_id', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
        most_common_age_bin=('age_bin', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'),
        unique_customers=('customer_id', 'nunique'),  # Count of unique customers
        unique_articles_sold=('article_id', 'nunique'),  # Count of unique articles sold
        median_age=('age', 'median'),            # Median age of customers
        fashion_news_subscribers=('fashion_news_frequency', lambda x: (x == 'Regularly').sum()),  # Count of subscribers
        first_purchase_days_ago=('elapsed_days_since_first_sell', 'min'),  # Days since the first purchase
        recent_purchase_days_ago=('elapsed_days_since_last_purchase', 'min'),  # Days since the last purchase
        product_type_name=('product_type_name', 'first'),
        colour_group_name=('colour_group_name', 'first'),
        graphical_appearance_name=('graphical_appearance_name', 'first')
    ).reset_index()

    return grouped_data


def get_top_product_groups(grouped_data, top_n=800):
    """
    Filters the top N most frequent product groups based on total transaction count.

    Parameters:
        grouped_data (pd.DataFrame): The grouped DataFrame containing product group data.
        top_n (int): The number of top product groups to select (default is 800).

    Returns:
        pd.DataFrame: A filtered DataFrame containing only the top N product groups.
    """
    # Aggregate transaction counts by product group
    product_group_totals = grouped_data.groupby('product_group')['transaction_count'].sum().reset_index()

    # Sort product groups by transaction count in descending order
    product_group_totals = product_group_totals.sort_values(by='transaction_count', ascending=False)

    # Select the top N product groups
    top_product_groups = product_group_totals.head(top_n)

    # Filter the original DataFrame to include only the top N product groups
    filtered_data = grouped_data[grouped_data['product_group'].isin(top_product_groups['product_group'])]

    return filtered_data

def load_best_hyperparameters(group_output_dir):
    """Loads the best hyperparameters from the CSV file if it exists."""
    best_params_path = os.path.join(group_output_dir, "best_hyperparameters.csv")
    if not os.path.exists(best_params_path):
        return None
    try:
        best_params = pd.read_csv(best_params_path).iloc[0].to_dict()
        return best_params
    except Exception as e:
        print(f"Error loading best hyperparameters for {group_output_dir}: {e}")
        return None

