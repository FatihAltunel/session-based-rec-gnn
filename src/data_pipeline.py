import pandas as pd
import numpy as np
import gc
from tqdm import tqdm

def load_and_preprocess_data(events_path, prop1_path, prop2_path):
    print("--- Step 1: Loading & Cleaning Data ---")
    events = pd.read_csv(events_path)

    # A. Timestamp Correction
    if events["timestamp"].max() > 10_000_000_000:
        events["timestamp"] = (events["timestamp"] // 1000).astype(int)

    # B. Noise Reduction (CRITICAL)
    item_counts = events['itemid'].value_counts()
    popular_items = item_counts[item_counts >= 5].index
    events = events[events['itemid'].isin(popular_items)].copy()
    print(f"Filtered Events: {len(events)} | Unique Items: {events['itemid'].nunique()}")

    # C. Event Type Mapping
    event_map = {'view': 1, 'addtocart': 2, 'transaction': 3}
    events['event_type'] = events['event'].map(event_map).fillna(1).astype(int)

    # D. Category Integration
    print("Loading Item Categories (Extracting 'categoryid' only)...")
    try:
        def get_categories(path):
            df = pd.read_csv(path)
            return df[df['property'] == 'categoryid'][['itemid', 'value']]

        cat1 = get_categories(prop1_path)
        cat2 = get_categories(prop2_path)
        cats = pd.concat([cat1, cat2])
        cats.rename(columns={'value': 'category_id'}, inplace=True)

        cats['category_id'] = pd.to_numeric(cats['category_id'], errors='coerce').fillna(0).astype(int)
        cats = cats.drop_duplicates(subset='itemid')

        events = events.merge(cats, on='itemid', how='left')
        events['category_id'] = events['category_id'].fillna(0).astype(int)
        print("✅ Categories merged successfully!")
        del cat1, cat2, cats
        gc.collect()
    except Exception as e:
        print(f"⚠️ Warning: Could not load categories ({e}). Proceeding without them.")
        events['category_id'] = 0

    # E. Sorting & Sessionizing
    print("Creating Sessions...")
    events = events.sort_values(by=["visitorid", "timestamp"])
    events['time_diff'] = events.groupby('visitorid')['timestamp'].diff()
    events['new_session'] = (events['visitorid'] != events['visitorid'].shift()) | (events['time_diff'] > 30*60)
    events['session_id'] = events['new_session'].cumsum()
    
    return events

def create_mappings_and_split(events):
    print("--- Step 2: Mappings & Splitting ---")
    all_items = events['itemid'].unique()
    item2idx = {item: i+1 for i, item in enumerate(all_items)}
    n_items = len(item2idx) + 1

    all_cats = events['category_id'].unique()
    cat2idx = {cat: i+1 for i, cat in enumerate(all_cats)}
    n_cats = len(cat2idx) + 1

    events['item_idx'] = events['itemid'].map(item2idx)
    events['cat_idx'] = events['category_id'].map(cat2idx)

    # Zamansal Bölme (7 gün Test, 7 gün Validation, Kalıbı Train)
    max_time = events['timestamp'].max()
    day_sec = 24 * 60 * 60
    test_start = max_time - (7 * day_sec)
    val_start = test_start - (7 * day_sec)

    train_df = events[events['timestamp'] < val_start]
    val_df = events[(events['timestamp'] >= val_start) & (events['timestamp'] < test_start)]
    test_df = events[events['timestamp'] >= test_start]

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df, n_items, n_cats

def process_sessions(df, augment=False):
    grouped = df.groupby('session_id')
    dataset = []

    for _, group in tqdm(grouped, desc=f"Processing (Augment={augment})"):
        items = group['item_idx'].tolist()
        cats = group['cat_idx'].tolist()
        evts = group['event_type'].tolist()

        if len(items) < 2: 
            continue

        if augment:
            # Sliding Window Veri Çoğaltması
            for i in range(1, len(items)):
                dataset.append({
                    'items': items[:i+1],
                    'cats': cats[:i+1],
                    'events': evts[:i+1]
                })
        else:
            dataset.append({
                'items': items,
                'cats': cats,
                'events': evts
            })
    return dataset