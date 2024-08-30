# Constants
tokens = 5.5e12  # 5.5 trillion tokens
flops_per_token = 20e9  # Rough estimate: 20 billion FLOPs per token for a 7B model
gpus = 64*4  # Number of H100 GPUs
gpu_performance = 1e15  # H100: 1,000 TFLOPs (1 petaflop) in half-precision

# Calculate total FLOPs required
total_flops = tokens * flops_per_token

# Calculate FLOPs per GPU
flops_per_gpu = total_flops / gpus

# Calculate time required in seconds per GPU
time_seconds = flops_per_gpu / gpu_performance

# Convert time to days
time_days = time_seconds / (3600 * 24)

# Output the estimated time
print(f"Estimated training time on {gpus} H100 GPUs: {time_days:.2f} days")

