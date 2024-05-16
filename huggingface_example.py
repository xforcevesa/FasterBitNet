from bitnet import replace_linears_in_hf

from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
head_model = GPT2LMHeadModel.from_pretrained('gpt2')

text = "Shanghai is great place to live, "
encoded_input = tokenizer(text, return_tensors='pt')
output = head_model.generate(**encoded_input)
output = tokenizer.decode(output[0], skip_special_tokens=True)
print("GPT2 output (before replacement of linears): ", output)

# Replace all linears in the model with bit-wise operations
replace_linears_in_hf(head_model)

text = "Shanghai is great place to live, "
encoded_input = tokenizer(text, return_tensors='pt')
output = head_model.generate(**encoded_input)
output = tokenizer.decode(output[0], skip_special_tokens=True)

print("GPT2 output (after replacement of linears): ", output)
