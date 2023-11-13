import torch


def mac_device_check():
    if torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def print_parameters(model):
    # model.state_dict() -> all parameter dictionaries
    # .items() extract every pair in the dictionary
    # such as (name1, value1)
    # for loop can read every element by iterations
    for name, weight in model.state_dict().items():
        print(name, ': ', weight)


def gen_data():
    return None
