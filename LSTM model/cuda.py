import torch

print(torch.cuda.is_available())  # Should print True if you have a working GPU install
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
