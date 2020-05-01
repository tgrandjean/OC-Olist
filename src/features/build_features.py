"""Transform raw dataset into model's features."""
import os
import pandas as pd

from src.features.categories import categ


def load_data(data_path):
    """Return dict with filenames as keys and file content as values."""
    data = dict()
    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:
            filepath = os.path.join(dirname, filename)
            data[filename.split('.')[0]] = pd.read_csv(filepath)
    return data


def customer_table(data):
    """Return datframe all customers."""
    customers = data['olist_customers_dataset'].copy()
    customers = customers[['customer_unique_id']].reset_index(drop=True)
    return customers


def customers_mapper(data):
    customers = data['olist_customers_dataset'].copy()
    customers = customers[['customer_unique_id', 'customer_id']]
    customers.set_index('customer_id', inplace=True, drop=True)
    return customers.to_dict()['customer_unique_id']


def orders_mapper(orders):
    return orders[['customer_unique_id', 'order_id']]\
           .set_index('order_id', drop=True)\
           .to_dict()['customer_unique_id']


def get_orders_between_two_dates(data, date_start, date_end,
                                 on="order_purchase_timestamp"):
    if (type(date_start) == str) and (type(date_end) == str):
        date_start, date_end = pd.to_datetime([date_start, date_end])
    orders = data['olist_orders_dataset']
    for col in orders.columns:
        if col.endswith('_id') or col.endswith('_status'):
            pass
        else:
            orders[col] = pd.to_datetime(orders[col])
    orders = orders[(orders[on] > date_start) & (orders[on] <= date_end)]
    return orders


def frequencies(customers, orders, data):
    mapper = customers_mapper(data)
    customers = customers.copy()
    orders = orders.copy()
    orders['customer_unique_id'] = orders['customer_id'].map(mapper)
    orders_count = orders[['customer_unique_id', 'order_id']]\
        .groupby('customer_unique_id').count()
    customers = customers.set_index('customer_unique_id')\
        .join(orders_count)
    customers.fillna(0, inplace=True)
    customers.rename(columns={"order_id": "frequency"}, inplace=True)
    return customers


def recencies(customers, orders, data):
    mapper = customers_mapper(data)
    customers = customers.copy()
    orders = orders.copy()
    orders['customer_unique_id'] = orders['customer_id'].map(mapper)
    if not orders.values.any():
        raise ValueError("No orders")
    last_command_for_customer = orders[['customer_unique_id',
                                        'order_purchase_timestamp']]\
        .sort_values('order_purchase_timestamp')\
        .groupby('customer_unique_id').last()
    recency = orders.order_purchase_timestamp.max() - last_command_for_customer
    customers = customers.join(recency)
    customers.rename(columns={'order_purchase_timestamp': 'recency'},
                     inplace=True)
    return customers


def monetary(customers, orders, data):
    mapper = customers_mapper(data)
    customers = customers.copy()
    orders = orders.copy()
    orders['customer_unique_id'] = orders['customer_id'].map(mapper)
    monetary = data['olist_order_items_dataset'].copy()
    monetary = monetary[['order_id', 'price']].groupby('order_id').sum()
    monetary.reset_index(inplace=True)
    mapper = orders_mapper(orders)
    monetary['customer_unique_id'] = monetary['order_id'].map(mapper)
    monetary = monetary.drop('order_id', axis=1)\
        .groupby('customer_unique_id').sum()
    customers = customers.join(monetary)
    customers['price'] = customers['price'].fillna(0)
    customers.rename(columns={'price': 'monetary'}, inplace=True)
    return customers


def items_per_cart(customers, orders, data):
    mapper = customers_mapper(data)
    customers = customers.copy()
    orders = orders.copy()
    orders['customer_unique_id'] = orders['customer_id'].map(mapper)
    items = data['olist_order_items_dataset'].copy()
    items = items.groupby('order_id').count()['order_item_id']
    items = items.reset_index(drop=False)
    mapper = orders_mapper(orders)
    items['customer_unique_id'] = items['order_id'].map(mapper)
    items = items.drop('order_id', axis=1)\
        .groupby('customer_unique_id').mean()
    customers = customers.join(items)
    customers['order_item_id'] = customers['order_item_id'].fillna(0)
    customers.rename(columns={'order_item_id': 'item_per_c'}, inplace=True)
    return customers


def monetary_per_categ(customers, orders, data):
    mapper = customers_mapper(data)
    customers = customers.copy()
    orders = orders.copy()
    orders['customer_unique_id'] = orders['customer_id'].map(mapper)
    items = data['olist_order_items_dataset'].copy()
    products = data['olist_products_dataset'].copy()
    translation = data['product_category_name_translation']\
        .set_index('product_category_name')\
        .to_dict()['product_category_name_english']
    products['category'] = products['product_category_name'].map(translation)
    products['mother_cat'] = products['category'].map(categ)
    categ_mapper = products[['product_id', 'mother_cat']]\
        .set_index('product_id').to_dict()['mother_cat']
    items['categ'] = items['product_id'].map(categ_mapper)
    items['customer'] = items['order_id'].map(orders_mapper(orders))
    table = pd.pivot_table(items, columns='categ', values='price',
                           index='customer')
    customers = customers.join(table)
    return customers


def reviews(customers, data):
    mapper = customers_mapper(data)
    customers = customers.copy()
    orders = data['olist_orders_dataset'].copy()
    orders['customer_unique_id'] = orders['customer_id'].map(mapper)
    reviews = data['olist_order_reviews_dataset'].copy()
    reviews['customer'] = reviews['order_id'].map(orders_mapper(orders))
    reviews = reviews.groupby('customer').mean()['review_score']
    customers = customers.join(reviews)
    return customers
