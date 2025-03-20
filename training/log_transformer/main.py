import src.utils.env
from src.dataset import LogDataset, DataLoader



def preprocess_dataset(examples):
    results = []
    for example in examples:
        result = []
        for s in example:
            result = str2int(s)
        result["attention_mask"] = [1] * len(result["input_ids"])
        result["labels"] = result["input_ids"].copy()
        results.append(result)
    return results

if __name__ == '__main__':

    dataset_ = LogDataset('data/Android_2k.log_structured.csv')
    print(len(dataset_))
    print(dataset_[0])

