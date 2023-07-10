import yaml
from torch import nn

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def read_model_config(config_path):
    with open(f"{config_path}", 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            return parsed_yaml
        except yaml.YAMLError as exc:
            raise exc

def load_weights(model, ckpt, strict=False):
    u, m = model.load_state_dict(ckpt, strict=strict)
    model_name = str(model.__class__).split('.')[-1].split("'")[0]
    print(f"Loading weights for {model_name}")
    if not (u or m):
        print(f"All keys matched")
    if u:
        print(f"Missing keys: {u}")
    if m:
        print(f"Unexpected keys: {m}")