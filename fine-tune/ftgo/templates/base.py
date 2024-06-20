

class BASE:
    def __init__(self):
        #self.tokenizer = tokenizer
        
        #self.max_token = max_token
        pass

    def config(self, tokenizer):
        self.tokenizer = tokenizer
        pass

    def preprocess(self, samples):
        raise NotImplementedError
    

    def truncate(self, text, max_seq_length):
        token = self.tokenizer.encode(text,
            add_special_tokens=False,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt"
        )[0]
        text = self.tokenizer.decode(token)
        return text



