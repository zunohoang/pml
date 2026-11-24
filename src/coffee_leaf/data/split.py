from sklearn.model_selection import train_test_split


def split_train_test(image_table, config):
    test_size = config["test_size"]
    seed = config["seed"]
    labels = image_table["label"]
    train_table, test_table = train_test_split(
        image_table,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    train_table = train_table.reset_index(drop=True)
    test_table = test_table.reset_index(drop=True)
    return train_table, test_table
