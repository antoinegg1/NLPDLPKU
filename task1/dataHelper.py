from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import json
import random

def get_texts_and_labels_restaurant(data_samples, sep_token):
    texts = [
        sample['sentence'].replace(sample['term'], f"{sep_token} {sample['term']}")
        for sample in data_samples.values()
    ]
    labels = [
        1 if sample['polarity'] == "positive" else
        2 if sample['polarity'] == "neutral" else
        0  
        for sample in data_samples.values()
    ]
    return texts, labels

def get_texts_and_labels_acl(data_samples):
    texts = [sample["text"] for sample in data_samples]
    unique_labels = sorted(set(sample['label'] for sample in data_samples))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    labels = [label_to_id[sample['label']] for sample in data_samples]
    return texts, labels

def get_restaurant_dataset(sep_token):
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
    # 加载 ag_news 数据集的测试集
    dataset = load_dataset('ag_news', split='test')
    # 使用 'description' 作为输入文本
    # 数据集已经将 'description' 存储在 'text' 字段中，无需处理
    # 使用 `train_test_split` 以 9:1 比例划分训练集和测试集，设定随机种子为 2022
    split_dataset = dataset.train_test_split(test_size=0.1, seed=2022)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']
    return train_dataset, test_dataset

def get_single_dataset(dataset_name, sep_token):
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
        raise ValueError(f"未识别的数据集名称：{dataset_name}")

    if second_name == "fs":
        train_dataset = sample_few_shot_dataset(train_dataset, seed=42)
        # test_dataset = sample_few_shot_dataset(test_dataset, seed=42)
    return train_dataset, test_dataset

def sample_few_shot_dataset(dataset, seed=42):
    # 设置随机种子
    random.seed(seed)
    # 获取所有唯一的标签
    labels = dataset['label']
    unique_labels = sorted(set(labels))
    num_labels = len(unique_labels)
    
    if num_labels < 5:
        # 标签数少于 5，每个类别尽量均衡，共 32 个样本
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
        # 如果总样本数不足 32，随机补充样本
        if len(selected_indices) < total_samples:
            remaining_indices = list(set(range(len(labels))) - set(selected_indices))
            random.shuffle(remaining_indices)
            selected_indices.extend(remaining_indices[:total_samples - len(selected_indices)])
    else:
        # 标签数大于或等于 5，每个类别随机选择 8 个样本
        K_per_label = 8
        indices_per_label = {label: [] for label in unique_labels}
        for idx, label in enumerate(labels):
            indices_per_label[label].append(idx)
        selected_indices = []
        for label in unique_labels:
            indices = indices_per_label[label]
            random.shuffle(indices)
            selected_indices.extend(indices[:min(K_per_label, len(indices))])
    # 打乱所有选中的样本索引
    random.shuffle(selected_indices)
    # 创建少样本数据集
    few_shot_dataset = dataset.select(selected_indices)
    return few_shot_dataset

def rearrange_labels(train_datasets, test_datasets):
    now_label_add_idx = 0
    output_train_datasets = []
    output_test_datasets = []
    for train_dataset, test_dataset in zip(train_datasets, test_datasets):
        # 获取训练集中所有唯一的标签
        unique_labels = sorted(set(train_dataset['label']))
        # 创建标签映射，避免标签重叠
        label_map = {old_label: new_label + now_label_add_idx for new_label, old_label in enumerate(unique_labels)}
        # 更新训练集和测试集的标签
        train_dataset = train_dataset.map(lambda example: {"label": label_map[example['label']]})
        test_dataset = test_dataset.map(lambda example: {"label": label_map[example['label']]})
        output_train_datasets.append(train_dataset)
        output_test_datasets.append(test_dataset)
        now_label_add_idx += len(unique_labels)
    return output_train_datasets, output_test_datasets

def get_dataset(dataset_name, sep_token):
    '''
    dataset_name: str 或 list of str，数据集的名称
    sep_token: str，tokenizer 使用的分隔符（例如 '<sep>'）
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
        # 重新排列标签，避免重叠
        train_datasets, test_datasets = rearrange_labels(train_datasets, test_datasets)
        # 合并数据集
        train_dataset = concatenate_datasets(train_datasets)
        test_dataset = concatenate_datasets(test_datasets)
    else:
        raise ValueError("dataset_name 应为字符串或字符串列表。")

    dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})
    return dataset

# # 测试代码
# if __name__ == '__main__':
#     # 调用 get_dataset 函数并加载数据集
#     dataset = get_dataset(['restaurant_fs', 'laptop_fs', 'acl_fs'], "<sep>")

#     # 检查数据集中是否包含 'train' 和 'test' 集合
#     assert 'train' in dataset, "Dataset does not contain a 'train' split."
#     assert 'test' in dataset, "Dataset does not contain a 'test' split."

#     # 检查 'train' 和 'test' 数据集的字段是否正确
#     for split in ['train', 'test']:
#         assert 'text' in dataset[split].column_names, f"'text' column is missing in the {split} dataset."
#         assert 'label' in dataset[split].column_names, f"'label' column is missing in the {split} dataset."

#     # 输出一些样本，检查 'sep_token' 是否已正确添加，以及情感标签是否符合预期
#     print("Sample from the train set:")
#     print("Text:", dataset['train']['text'][0])
#     print("Label:", dataset['train']['label'][0])

#     print("\nSample from the test set:")
#     print("Text:", dataset['test']['text'][0])
#     print("Label:", dataset['test']['label'][0])

#     # 输出标签范围
#     print("\nLabel range in train set:", set(dataset['train']['label']))
#     print("Label range in test set:", set(dataset['test']['label']))
#     # 输出数据集大小
#     print("\nTrain set size:", len(dataset['train']))
#     print("Test set size:", len(dataset['test']))

#     print("\nDataset is processed successfully.")

