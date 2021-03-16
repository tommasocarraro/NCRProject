from ncr.data import Dataset

if __name__ == '__main__':
    #dataset = Dataset("datasets/u.data")
    dataset = Dataset("datasets/fake.csv", sep=',', convert_to_indexes=True)

    dataset.process_data(threshold=4, order=True)

    print("Dataset before preprocessing")
    print(dataset.dataset)

    print("Dataset after preprocessing")
    print(dataset.proc_dataset)

    print("Train set interactions")
    print(dataset.train)

    print("Validation set interactions")
    print(dataset.validation)

    print("Test set interactions")
    print(dataset.test)
