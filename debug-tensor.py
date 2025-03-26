import torch

dir = "/home/chenyangli_google_com/vllm/tensor_save/"

# logits  = torch.load(dir + "first_logits.pt")
# mask = torch.load(dir + "first_grammar_bitmask.pt")
# masked_logits = torch.load(dir + "first_masked_logits.pt", weights_only=True)
# print(logits.shape, mask.shape, masked_logits.shape)
# print(mask[mask!=0].shape, mask[mask!=0])
# print(masked_logits[masked_logits!=-float('inf')].shape, masked_logits[masked_logits!=-float('inf')])
# selected_logits = torch.argmax(masked_logits, dim=-1, keepdim=True)
# print(selected_logits)
# # find out which index in masked_logits is not -inf
# non_inf_indices = torch.nonzero(masked_logits!=-float('inf'))
# print('non_inf_indices: ', non_inf_indices)
# print(masked_logits[0][90], masked_logits[0][4913])


# 0: {"
masked_logits = torch.load(dir + "masked_logits_0.pt")
print("logits valid bits: ", masked_logits[masked_logits!=-float('inf')].shape, masked_logits[masked_logits!=-float('inf')])
selected_logits = torch.argmax(masked_logits, dim=-1, keepdim=True)
print(selected_logits)
# 1: name
masked_logits = torch.load(dir + "masked_logits_1.pt")
print("logits valid bits: ", masked_logits[masked_logits!=-float('inf')].shape, masked_logits[masked_logits!=-float('inf')])
selected_logits = torch.argmax(masked_logits, dim=-1, keepdim=True)
print(selected_logits)
exit()
# 8: should be age instead of a + ge(9)
masked_logits = torch.load(dir + "masked_logits_8.pt")
print(masked_logits.shape)
print("logits valid bits: ", masked_logits[masked_logits!=-float('inf')].shape, masked_logits[masked_logits!=-float('inf')])
selected_logits = torch.argmax(masked_logits, dim=-1, keepdim=True)
print(selected_logits)
print(f"score for a: {masked_logits[0][56693]}, score for age: {masked_logits[0][424]}")
