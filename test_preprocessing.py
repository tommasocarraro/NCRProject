from ncr.data import Dataset
import pandas as pd

if __name__ == '__main__':
    raw_dataset = pd.read_csv("datasets/toy-dataset/fake.csv")
    dataset = Dataset(raw_dataset)

    dataset.process_data(threshold=4, order=True, leave_n=1, keep_n=5, max_history_length=5, premise_threshold=0)

    print("Dataset before preprocessing")
    print(dataset.dataset)

    print("Dataset after preprocessing")
    print(dataset.proc_dataset)

    print("Train set interactions")
    print(dataset.train_set)

    print("Validation set interactions")
    print(dataset.validation_set)

    print("Test set interactions")
    print(dataset.test_set)
