"""Check CUDA, bf16, and an actual GPU operation before downloading the model."""

def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install the CUDA PyTorch wheel and check nvidia-smi")
    print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    print("Compiled architectures:", torch.cuda.get_arch_list())
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bf16")
    matrix = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    result = matrix @ matrix.T
    torch.cuda.synchronize()
    if not torch.isfinite(result).all():
        raise RuntimeError("GPU bf16 matrix multiplication failed")
    print("CUDA/bf16 execution check passed")


if __name__ == "__main__":
    main()
