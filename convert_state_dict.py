import sys
import torch

if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
    print("Run script with 1 argument as follows: 'python3 convert_state_dict.py <filepath>'")
    exit(0)
state_path = sys.argv[1]
two_models = (len(sys.argv) == 3 and int(sys.argv[2]) == 2)

new_pathname = state_path.split('.')
new_pathname = '.'.join(new_pathname[:-1]) + "_converted." + new_pathname[-1]
model_state = torch.load(state_path, map_location='cpu')['sd']
new_model_state = {}

for name, value in model_state.items():
    model = "model_2" if two_models and name.split('.')[0] == "receiver" else "model"
    name = "sent_encoder._text_field_embedder." + model + "." + '.'.join(name.split('.')[1:])
    name_parts = name.split('.')
    if name_parts[-1] == 'gamma':
        name = '.'.join(name_parts[:-1]) + ".weight"
    elif name_parts[-1] == 'beta':
        name = '.'.join(name_parts[:-1]) + ".bias"
    new_model_state[name] = value

torch.save(new_model_state, new_pathname)
