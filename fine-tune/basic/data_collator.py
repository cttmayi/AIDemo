from trl import DataCollatorForCompletionOnlyLM

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

instruction_template = "###Inst:"
response_template = "###Resp:"



text = "###Inst: AB\n###Resp: HI\n###Inst: AB\n###Resp: HI\n"
text_ids = tokenizer(text)
print('text_ids')
print(text_ids)

print('tokenize(text)')
print(tokenizer.tokenize(text))

data_collator = DataCollatorForCompletionOnlyLM(
    # instruction_template=instruction_template,
    response_template=response_template, 
    tokenizer=tokenizer,
    )


batch = data_collator([text_ids])

print('tbatch')
print(batch)
print([text_ids])

print('batch')
for key in batch.keys():
    print(key, batch[key])