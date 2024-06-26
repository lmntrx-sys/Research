import torch
import torch.nn as nn

eps = 1e-5

def binarize_weights(self):
   alpha = self.weight.mean()
   binarized_weights = torch.sign(self.weight - alpha)
   return binarized_weights

def quantize_activations(self, x, b=8):
   Q_b = 2 ** (b - 1)
   gamma = x.abs().max()
   quantized_x = torch.clamp(
      x * Q_b / (gamma + eps), -Q_b + eps, Q_b - eps
  )
   return quantized_x

def scale_activations(self, x, b=8):
   Q_b = 2 ** (b - 1)
   eta = x.min()
   gamma = x.abs().max()
   scaled_x = torch.clamp(
      (x - eta) * Q_b / (gamma + eps), eps, Q_b - eps
   )
   return scaled_x

import torch
import torch.nn as nn

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, dtype=None, num_groups=1):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.dtype = dtype if dtype is not None else torch.bfloat16
        self.num_groups = num_groups
        self.eps = 1e-5  # Small epsilon value to avoid division by zero and overflow

        # Store original weights
        self.original_weights = self.weight.data.clone().detach()
        
        # Initialize 1-bit quantized weights and store them as int8
        self.register_buffer(
            "quantized_weights", torch.sign(self.weight.data).to(torch.int8)
        )
        # Clear the original weights to save memory
        del self._parameters['weight']

    @property
    def weight(self):
        # Return the dequantized weights when accessed
        return self.dequantize_weights()

    @weight.setter
    def weight(self, value):
        # Update the quantized_weights when the weight property is set
        self.quantized_weights.data = torch.sign(value).to(torch.int8)
        # Also update the original weights
        self.original_weights = value.clone().detach()

    def ste_binarize(self, x):
        # Find the alpha which is the mean
        binarized = torch.sign(x)
        output = (binarized - x).detach() + x 
        return output

    def dequantize_weights(self):
        # Convert quantized_weights back to float32 and compute alpha for the weights
        float_weights = self.quantized_weights.to(torch.float32)
        alpha = float_weights.abs().mean()
        return float_weights * alpha

    def binarize_weights_groupwise(self):
        # Dequantize the weights before binarization
        weights = self.dequantize_weights().to(self.dtype)

        # Divide weights into groups
        group_size = weights.shape[0] // self.num_groups
        binarized_weights = torch.zeros_like(weights, dtype=self.dtype)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = weights[start_idx:end_idx]

            # Binarize each group using STE
            alpha_g = weight_group.abs().mean()
            binarized_weights[start_idx:end_idx] = self.ste_binarize(
                weight_group - alpha_g
            )

        return binarized_weights

    def quantize_activations_groupwise(self, x, b=8):
        Q_b = 2 ** (b - 1)

        # Divide activations into groups
        group_size = x.shape[0] // self.num_groups
        quantized_x = torch.zeros_like(x)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            activation_group = x[start_idx:end_idx]

            # Quantize each group
            gamma_g = activation_group.abs().max()
            quantized_x[start_idx:end_idx] = torch.clamp(
                activation_group * Q_b / (gamma_g + self.eps),
                -Q_b + self.eps,
                Q_b - self.eps,
            )

        return quantized_x

    def forward(self, input: torch.Tensor):
        # Ensure the input tensor is in the correct dtype
        input = input.to(self.dtype)
        
        # Binarize weights (group-wise) using STE
        binarized_weights = self.binarize_weights_groupwise()
        
        # Apply linear transformation with binarized weights
        output = nn.functional.linear(input, binarized_weights, self.bias.to(self.dtype))
        
        # Quantize activations (ensure the method returns the quantized output)
        output = self.quantize_activations_groupwise(output)
        
        return output

    def get_original_weights(self):
        return self.original_weights

class SimpleLinear(nn.Module):
    def __init__(self, input_features, out_features, num_groups):
        super(SimpleLinear, self).__init__()
        self.linear = BitLinear(input_features, out_features, num_groups=num_groups)
        
    def forward(self, x):
        return self.linear(x)

# Example usage
input_shape = 64
out_shape = 128
sample_size = 200

inputs = torch.randn(sample_size, input_shape)
true_weights = torch.randn(input_shape, out_shape)

# Generating synthetic targets: y = inputs x true_weights + noise
noise = 0.05 * torch.randn(sample_size, out_shape)
targets = inputs @ true_weights + noise

inputs = inputs.to(torch.bfloat16)
targets = targets.to(torch.bfloat16)

print(inputs.dtype, targets.dtype)

model = SimpleLinear(input_shape, out_shape, num_groups=5)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 500
losses = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    
    # Convert outputs and targets to float32 for the loss computation
    loss = criterion(outputs.to(torch.float32), targets.to(torch.float32))
    
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

# Retrieve original weights
original_weights = model.linear.get_original_weights()
print("Original Weights:", original_weights)
print(model.linear.weight)
