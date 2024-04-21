from torch.utils.data import Dataset, DataLoader
from preprocessor import tokenize, yield_sentence_pair, yield_lines, load_preprocessor, read_lines, \
    count_line, OUTPUT_DIR, get_complexity_score, safe_division, get_word2rank, remove_stopwords, remove_punctuation
from preprocessor import  get_data_filepath


##### build dataset Loader #####
class TrainDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len=256, sample_size=1):
        self.sample_size = sample_size
        print("init TrainDataset ...")
        self.source_filepath = get_data_filepath(dataset,'train','complex')
        self.target_filepath = get_data_filepath(dataset,'train','simple')
        print("Initialized dataset done.....")
        # preprocessor = load_preprocessor()
        # self.source_filepath = preprocessor.get_preprocessed_filepath(dataset, 'train', 'complex')
        # self.target_filepath = preprocessor.get_preprocessed_filepath(dataset, 'train', 'simple')

        self.max_len = max_len
        self.tokenizer = tokenizer

        self._load_data()

    def _load_data(self):
        self.inputs = read_lines(self.source_filepath)
        self.targets = read_lines(self.target_filepath)

    def __len__(self):
        return int(len(self.inputs) * self.sample_size)

    def __getitem__(self, index):
        source = self.inputs[index]
        # source = "summarize: " + self.inputs[index]
        target = self.targets[index]

        tokenized_inputs = self.tokenizer(
            [source],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors="pt"
        )
        tokenized_targets = self.tokenizer(
            [target],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors="pt"
        )
        source_ids = tokenized_inputs["input_ids"].squeeze()
        target_ids = tokenized_targets["input_ids"].squeeze()

        src_mask = tokenized_inputs["attention_mask"].squeeze()  # might need to squeeze
        target_mask = tokenized_targets["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask,
                'sources': self.inputs[index], 'targets': [self.targets[index]],
                'source': source, 'target': target}


class ValDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len=256, sample_size=1):
        self.sample_size = sample_size
        ### WIKI-large dataset ###
        self.source_filepath = get_data_filepath(dataset, 'valid', 'complex')
        self.target_filepaths = get_data_filepath(dataset, 'valid', 'simple')

        ### turkcorpus dataset ###
        # self.source_filepath = get_data_filepath(TURKCORPUS_DATASET, 'valid', 'complex')
        # self.target_filepaths = [get_data_filepath(TURKCORPUS_DATASET, 'valid', 'simple.turk',i)for i in range(8)]
        # if dataset == NEWSELA_DATASET:
        #     self.target_filepaths = [get_data_filepath(dataset, 'valid', 'simple')]

        # else:  # TURKCORPUS_DATASET as default
        #     self.target_filepaths = [get_data_filepath(TURKCORPUS_DATASET, 'valid', 'simple.turk', i) for i in range(8)]

        self.max_len = max_len
        self.tokenizer = tokenizer

        self._build()

    def __len__(self):
        return int(len(self.inputs) * self.sample_size)

    def __getitem__(self, index):
        return {"source": self.inputs[index], "targets": self.targets[index]}

    def _build(self):
        self.inputs = []
        self.targets = []

        for source in yield_lines(self.source_filepath):
            self.inputs.append(source)
        
        for target in yield_lines(self.target_filepaths):
            self.targets.append(target)

        ### turkcorpus dataset ###
        # self.targets = [ [] for _ in range(count_line(self.target_filepaths[0]))]
        # for file_path in self.target_filepaths:
        #     for i, target in enumerate(yield_lines(file_path)):
        #         self.targets[i].append(target)

        # self.targets = [[] for _ in range(count_line(self.target_filepaths[0]))]
        # for filepath in self.target_filepaths:
        #     for idx, line in enumerate(yield_lines(filepath)):
        #         self.targets[idx].append(line)
