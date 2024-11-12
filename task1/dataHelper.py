from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import json
import random

def get_texts_and_labels_restaurant(data_samples, sep_token):
    # Process each sample to insert the separator token around terms, and get sentiment labels
    texts = [
        sample['sentence'].replace(sample['term'], f"{sep_token} {sample['term']}")
        for sample in data_samples.values()
    ]
    labels = [
        1 if sample['polarity'] == "positive" else
        2 if sample['polarity'] == "neutral" else
        0  # sentiment: positive=1, neutral=2, negative=0
        for sample in data_samples.values()
    ]
    return texts, labels

def get_texts_and_labels_acl(data_samples):
    # Extract texts and map labels to numerical values
    texts = [sample["text"] for sample in data_samples]
    unique_labels = sorted(set(sample['label'] for sample in data_samples))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    labels = [label_to_id[sample['label']] for sample in data_samples]
    return texts, labels

def get_restaurant_dataset(sep_token):
    # Load restaurant review dataset and process for training and testing
    train_dir = "./SemEval14-res/train.json"
    test_dir = "./SemEval14-res/test.json"
    with open(train_dir, 'r') as f:
        train_data = json.load(f)
    with open(test_dir, 'r') as f:
        test_data = json.load(f)
    train_texts, train_labels = get_texts_and_labels_restaurant(train_data, sep_token)
    test_texts, test_labels = get_texts_and_labels_restaurant(test_data, sep_token)
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})
    return train_dataset, test_dataset

def get_laptop_dataset(sep_token):
    # Load laptop review dataset and process for training and testing
    train_dir = "./SemEval14-laptop/train.json"
    test_dir = "./SemEval14-laptop/test.json"
    with open(train_dir, 'r') as f:
        train_data = json.load(f)
    with open(test_dir, 'r') as f:
        test_data = json.load(f)
    train_texts, train_labels = get_texts_and_labels_restaurant(train_data, sep_token)
    test_texts, test_labels = get_texts_and_labels_restaurant(test_data, sep_token)
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})
    return train_dataset, test_dataset

def get_acl_dataset():
    # Load ACL dataset for training and testing
    train_dir = "./ACL/train.jsonl"
    test_dir = "./ACL/test.jsonl"
    with open(train_dir, 'r') as f:
        train_data = [json.loads(line) for line in f]
    with open(test_dir, 'r') as f:
        test_data = [json.loads(line) for line in f]
    train_texts, train_labels = get_texts_and_labels_acl(train_data)
    test_texts, test_labels = get_texts_and_labels_acl(test_data)
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
    test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})
    return train_dataset, test_dataset

def get_agnews_dataset():
    # Load the AG News dataset, using a 9:1 train-test split with seed for consistency
    dataset = load_dataset('ag_news', split='test')
    split_dataset = dataset.train_test_split(test_size=0.1, seed=2022)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']
    return train_dataset, test_dataset

def get_single_dataset(dataset_name, sep_token):
    # Determine the specific dataset and type (e.g., full set or few-shot) based on name
    first_name, second_name = dataset_name.split("_")[0], dataset_name.split("_")[1]

    if first_name == "acl":
        train_dataset, test_dataset = get_acl_dataset()
    elif first_name == "agnews":
        train_dataset, test_dataset = get_agnews_dataset()
    elif first_name == "restaurant":
        train_dataset, test_dataset = get_restaurant_dataset(sep_token)
    elif first_name == "laptop":
        train_dataset, test_dataset = get_laptop_dataset(sep_token)
    else:
        raise ValueError(f"Unrecognized dataset name: {dataset_name}")

    if second_name == "fs":
        train_dataset = sample_few_shot_dataset(train_dataset, seed=42)
    return train_dataset, test_dataset

def sample_few_shot_dataset(dataset, seed=42):
    # Sample a few-shot dataset from the original, balanced across labels if possible
    random.seed(seed)
    labels = dataset['label']
    unique_labels = sorted(set(labels))
    num_labels = len(unique_labels)
    
    if num_labels < 5:
        # For fewer labels, balance across all and aim for 32 samples in total
        total_samples = 32
        K_per_label = total_samples // num_labels
        indices_per_label = {label: [] for label in unique_labels}
        for idx, label in enumerate(labels):
            indices_per_label[label].append(idx)
        selected_indices = []
        for label in unique_labels:
            indices = indices_per_label[label]
            random.shuffle(indices)
            selected_indices.extend(indices[:min(K_per_label, len(indices))])
        # Fill up to 32 samples if below the target number
        if len(selected_indices) < total_samples:
            remaining_indices = list(set(range(len(labels))) - set(selected_indices))
            random.shuffle(remaining_indices)
            selected_indices.extend(remaining_indices[:total_samples - len(selected_indices)])
    else:
        # For more labels, sample 8 instances per label
        K_per_label = 8
        indices_per_label = {label: [] for label in unique_labels}
        for idx, label in enumerate(labels):
            indices_per_label[label].append(idx)
        selected_indices = []
        for label in unique_labels:
            indices = indices_per_label[label]
            random.shuffle(indices)
            selected_indices.extend(indices[:min(K_per_label, len(indices))])
    random.shuffle(selected_indices)
    few_shot_dataset = dataset.select(selected_indices)
    return few_shot_dataset

def rearrange_labels(train_datasets, test_datasets):
    # Adjust label indices to avoid overlap across multiple datasets
    now_label_add_idx = 0
    output_train_datasets = []
    output_test_datasets = []
    for train_dataset, test_dataset in zip(train_datasets, test_datasets):
        unique_labels = sorted(set(train_dataset['label']))
        label_map = {old_label: new_label + now_label_add_idx for new_label, old_label in enumerate(unique_labels)}
        train_dataset = train_dataset.map(lambda example: {"label": label_map[example['label']]})
        test_dataset = test_dataset.map(lambda example: {"label": label_map[example['label']]})
        output_train_datasets.append(train_dataset)
        output_test_datasets.append(test_dataset)
        now_label_add_idx += len(unique_labels)
    return output_train_datasets, output_test_datasets

def get_dataset(dataset_name, sep_token):
    '''
    dataset_name: str or list of str, the name of the dataset
    sep_token: str, separator token used by the tokenizer (e.g., '<sep>')
    '''
    if isinstance(dataset_name, str):
        train_dataset, test_dataset = get_single_dataset(dataset_name, sep_token)
    elif isinstance(dataset_name, list):
        train_datasets= []
        test_datasets = []
        for name in dataset_name:
            train_ds, test_ds = get_single_dataset(name, sep_token)
            train_datasets.append(train_ds)
            test_datasets.append(test_ds)
        train_datasets, test_datasets = rearrange_labels(train_datasets, test_datasets)
        train_dataset = concatenate_datasets(train_datasets)
        test_dataset = concatenate_datasets(test_datasets)
    else:
        raise ValueError("dataset_name should be a string or a list of strings.")

    dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})
    return dataset
