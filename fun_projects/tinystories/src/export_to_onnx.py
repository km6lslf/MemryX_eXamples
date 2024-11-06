from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn

tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")

model.eval()

max_length = 128

dummy_input = tokenizer("Once upon a time", return_tensors="pt").input_ids

dummy_input = torch.cat([dummy_input, torch.zeros(1, max_length - dummy_input.size(1), dtype=torch.long)], dim=1)

class WrappedModel(nn.Module):
    def __init__(self, original_model):
        super(WrappedModel, self).__init__()
        self.original_model = original_model
    def forward(self, x):
        out = self.original_model(x)
        return out[0]

wrapped_model = WrappedModel(model)
wrapped_model.eval()


if os.path.exists("../models"):
    exportfilepath = "../models/tinystories33M.onnx"
elif os.path.exists("models"):
    exportfilepathpath = "models/tinystories33M.onnx"
else:
    exportfilepath = "tinystories33M.onnx"

torch.onnx.export(wrapped_model, dummy_input,
    exportfilepath, input_names=["input_ids"],
    output_names=["output"], dynamic_axes=None
)
