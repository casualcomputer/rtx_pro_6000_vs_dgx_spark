# LLM Inference Performance Comparison: NVIDIA DGX Spark (GB10) vs RTX Pro 6000 (Blackwell)

## Summary
This guide compares real-world LLM inference performance between NVIDIA's DGX Spark (GB10) integrated system and the RTX Pro 6000 Blackwell workstation GPU. Based on comprehensive benchmarks, the RTX Pro 6000 delivers **6-7x faster inference** across all batch sizes and model types, with significantly lower end-to-end latency.

<img width="1346" height="754" alt="rtx_pro_6000_vs_dgx_spark" src="https://github.com/user-attachments/assets/a388480c-cbef-4137-acd8-8d723baa5ff0" />

---

## Understanding the Hardware

When evaluating hardware for LLM deployment, it's important to understand the architectural differences that impact performance.

| Specification | DGX Spark (GB10) | RTX Pro 6000 (Blackwell) |
|---------------|------------------|--------------------------|
| **Architecture** | Grace Blackwell integrated system | Blackwell GPU workstation card |
| **CPU** | 20-core Arm (10× Cortex-X925 + 10× Cortex-A725) | — |
| **GPU** | Blackwell GPU, 5th-gen Tensor Cores | Blackwell GPU, 5th-gen Tensor Cores, 4th-gen RT Cores |
| **CUDA Cores** | — | 24,064 |
| **Memory Type** | LPDDR5X unified (CPU-GPU shared) | GDDR7 with ECC (dedicated) |
| **Memory Capacity** | 128 GB | 96 GB |
| **Memory Bandwidth** | 273 GB/s | 1,792 GB/s |
| **Memory Interface** | 256-bit | 512-bit |
| **Power** | 240W (system) | 600W (board) |
| **Form Factor** | 150 × 150 × 50.5 mm (~1.2 kg) | Dual-slot PCIe (5.4″ H × 12″ L) |
| **Key Feature** | Unified memory architecture | High-bandwidth dedicated GPU memory |

## What Matters for LLM Inference Performance

LLM inference performance is determined by three key metrics:

1. **End-to-end (E2E) latency**: Total time from request to complete response
2. **Prefill throughput**: How quickly the model processes input context (measured in tokens/second)
3. **Decode throughput**: How quickly the model generates output tokens (measured in tokens/second)

Understanding the difference between prefill and decode is important: prefill happens once per request to process the input, while decode happens iteratively for each output token. For a typical conversation with 2048 input tokens and 2048 output tokens, you prefill once and decode 2048 times.

### End-to-End Latency Formula

The end-to-end latency for a single request is calculated as:

```
E2E Latency (seconds) = (Input Length / Prefill Throughput) + ((Output Length × Batch Size) / Decode Throughput)
```

**Example calculation** for Llama 3.1 8B at batch size 1:

**DGX Spark:**
- Prefill throughput: 7,991 tokens/sec
- Decode throughput: 20.52 tokens/sec
- E2E = (2048 / 7,991) + ((2048 × 1) / 20.52)
- E2E = 0.256 + 99.805 = **100.1 seconds**

**RTX Pro 6000:**
- Prefill throughput: 38,744 tokens/sec
- Decode throughput: 143.90 tokens/sec
- E2E = (2048 / 38,744) + ((2048 × 1) / 143.90)
- E2E = 0.053 + 14.232 = **14.3 seconds**

This formula shows why decode throughput dominates overall latency: for long outputs, the decode phase (which happens 2048 times) takes much longer than the prefill phase (which happens once).

## Benchmark Configuration

**Test Parameters:**
- **Precision**: FP8 (standard for production inference)
- **Framework**: SGLang inference engine
- **Context**: 2048 input tokens / 2048 output tokens
- **Models tested**: 
  - Deepseek R1 14B
  - Gemma 3 12B
  - Gemma 3 27B
  - Llama 3.1 8B
  - Llama 3.1 70B
  - Qwen 3 32B
- **Batch sizes**: 1, 2, 4, 8, 16, 32 (representing single-user to multi-tenant scenarios)

*Benchmark data source: [LMSYS.org study](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)*

## Performance Results

### Overall Performance Across All Models

The table below shows the average speedup of RTX Pro 6000 compared to DGX Spark across all tested models:

| Batch Size | Prefill Speedup | Decode Speedup | E2E Speedup |
|------------|----------------|----------------|-------------|
| 1          | 7.07x          | 6.18x          | **6.18x**   |
| 2          | 6.80x          | 5.92x          | **5.93x**   |
| 4          | 6.20x          | 6.06x          | **6.06x**   |
| 8          | 6.03x          | 6.07x          | **6.07x**   |
| 16         | 6.16x          | 6.08x          | **6.08x**   |
| 32         | 4.79x          | 7.01x          | **7.00x**   |

**Key observation**: RTX Pro 6000 maintains approximately **6x faster** performance across all batch sizes, from single-user (batch 1) to high-concurrency (batch 32) scenarios.

### Example: Llama 3.1 8B Performance

Llama 3.1 8B is one of the most commonly deployed models. Here's how the two systems compare:

**Batch Size 1 (Single User):**
- DGX Spark: 100.1 seconds end-to-end
- RTX Pro 6000: 14.3 seconds end-to-end
- **Speedup**: 7.0x faster

**Batch Size 8 (Concurrent Users):**
- DGX Spark: 114.1 seconds per request
- RTX Pro 6000: 17.0 seconds per request
- **Speedup**: 6.7x faster

### Example: Llama 3.1 70B Performance

For larger models that utilize more memory:

**Batch Size 1:**
- DGX Spark: 772 seconds
- RTX Pro 6000: 100 seconds
- **Speedup**: 7.7x faster

**Batch Size 8:**
- DGX Spark: 813 seconds
- RTX Pro 6000: 107 seconds
- **Speedup**: 7.6x faster

### Performance Across Different Model Sizes (Batch Size 1)

| Model | Size | DGX Spark E2E (s) | RTX Pro 6000 E2E (s) | Speedup |
|-------|------|-------------------|----------------------|---------|
| Llama 3.1 | 8B | 100.1 | 14.3 | 7.0x |
| Deepseek R1 | 14B | 171.3 | 28.0 | 6.1x |
| Gemma 3 | 12B | 301.0 | 46.5 | 6.5x |
| Gemma 3 | 27B | 537.6 | 90.9 | 5.9x |
| Qwen 3 | 32B | 338.6 | 87.8 | 3.9x |
| Llama 3.1 | 70B | 772.5 | 100.0 | 7.7x |

## Understanding the Performance Difference

The 6x performance gap comes down to fundamental hardware characteristics that affect LLM inference workloads.

### Memory Bandwidth: The Primary Factor

LLM inference is primarily memory-bound rather than compute-bound. During token generation, the model weights must be loaded from memory for each forward pass. The speed at which this happens determines overall performance.

**Memory Bandwidth Comparison:**
- DGX Spark: 273 GB/s (LPDDR5X)
- RTX Pro 6000: 1,792 GB/s (GDDR7)
- **Bandwidth ratio**: 6.57x

The 6.57x memory bandwidth advantage closely corresponds to the ~6x inference speedup observed across benchmarks. This demonstrates that memory bandwidth is the limiting factor for these workloads.

### Memory Architecture Differences

**DGX Spark's Unified Memory:**
- CPU and GPU share the same 128GB LPDDR5X pool
- Enables zero-copy data sharing between CPU and GPU
- Lower bandwidth compared to dedicated GPU memory
- Beneficial for workloads requiring frequent CPU-GPU data movement

**RTX Pro 6000's Dedicated Memory:**
- 96GB GDDR7 dedicated to GPU
- Higher bandwidth optimized for GPU operations
- Requires explicit data transfers between CPU and GPU
- Optimized for GPU-intensive workloads

For LLM inference, the model weights typically remain on the GPU throughout inference, minimizing the benefit of unified memory while the performance is constrained by the lower bandwidth.

### Compute Performance

While both systems feature Blackwell GPUs with 5th-generation Tensor Cores supporting FP4/FP8 operations, compute throughput is not the bottleneck for these workloads. The limiting factor is how quickly model weights can be fed to the compute units from memory.

## Performance Scaling Across Batch Sizes

An important observation from the benchmarks: the performance ratio remains consistent across different batch sizes.

| Batch Size | E2E Speedup |
|------------|-------------|
| 1          | 6.18x       |
| 2          | 5.93x       |
| 4          | 6.06x       |
| 8          | 6.07x       |
| 16         | 6.08x       |
| 32         | 7.00x       |

This consistency indicates that:
1. Both systems scale their batch processing reasonably well
2. The memory bandwidth limitation persists regardless of concurrency
3. The RTX Pro 6000's bandwidth advantage isn't saturated even at higher batch sizes

## Memory Capacity Considerations

**Maximum Model Sizes (FP8 precision, approximate):**

**Models that fit on both systems:**
- Up to ~70B parameters (requires ~75GB)
- Examples: Llama 3.1 70B, Mistral Large, Mixtral 8x22B

**Models requiring more memory:**
- 200B+ parameter models require distributed inference
- Neither system can run Llama 3.1 405B (~405GB) locally

**Practical consideration**: The DGX Spark's additional 32GB of memory (128GB vs 96GB) extends the range of models that can fit, but for the majority of commonly deployed models (8B to 70B parameters), both systems have sufficient capacity. The performance difference is the more significant factor for these common use cases.

## Throughput and Capacity Planning

The performance difference directly impacts how many requests each system can handle.

**Example calculation** (Llama 3.1 8B, single request processing):

**DGX Spark:**
- Processing time: 100 seconds/request
- Theoretical daily capacity: ~864 requests (24 hrs × 3600 sec ÷ 100 sec)

**RTX Pro 6000:**
- Processing time: 14 seconds/request
- Theoretical daily capacity: ~6,171 requests

**Capacity ratio**: RTX Pro 6000 can handle approximately 7x more requests per day than DGX Spark for this model.

This means for serving the same number of requests, you would need multiple DGX Spark units to match a single RTX Pro 6000's throughput, affecting:
- Hardware costs
- Power consumption (240W × N units vs 600W × 1 unit)
- Rack space requirements
- System management complexity

## Use Case Considerations

Each system has distinct characteristics that may make it more suitable for specific scenarios:

### DGX Spark (GB10) Advantages:
- **Form factor**: Extremely compact (150×150×50mm) for edge deployment
- **Power efficiency**: 240W system power for constrained environments
- **Unified memory**: Beneficial for workloads requiring CPU-GPU data sharing
- **Integrated design**: All components in a single package
- **Networking**: Built-in 200Gb/s networking and Wi-Fi 7

**Best suited for**:
- Edge AI deployments with power/space constraints
- Applications requiring tight CPU-GPU integration
- Mobile or remote installations
- Workloads that benefit from unified memory architecture

### RTX Pro 6000 (Blackwell) Advantages:
- **Memory bandwidth**: 6.5x higher (1,792 GB/s vs 273 GB/s)
- **Inference performance**: 6-7x faster across all tested scenarios
- **Throughput**: Significantly higher request capacity
- **Enterprise features**: Multi-instance GPU (MIG) support, advanced video encoding

**Best suited for**:
- Data center LLM inference deployments
- High-throughput serving applications
- Multi-tenant SaaS platforms
- Scenarios where latency is critical
- Workloads requiring maximum inference speed

## Key Takeaways

1. **Performance**: RTX Pro 6000 delivers approximately 6-7x faster LLM inference across all batch sizes and models tested

2. **Memory bandwidth** is the primary determinant of LLM inference performance, where RTX Pro 6000's 1,792 GB/s significantly outperforms DGX Spark's 273 GB/s

3. **Consistent scaling**: The performance advantage remains stable from single-user (batch 1) to high-concurrency (batch 32) workloads

4. **Memory capacity**: Both systems can handle commonly deployed models up to 70B parameters. DGX Spark's additional 32GB provides headroom for slightly larger models

5. **Use case dependent**: Choice between systems should be based on specific requirements:
   - **For throughput-critical applications**: RTX Pro 6000's performance advantage is substantial
   - **For power/space-constrained deployments**: DGX Spark's compact form factor and lower power consumption are beneficial

## Reference Data

All performance data in this comparison is sourced from LMSYS.org's comprehensive benchmark study published in October 2025. The study tested both systems under identical conditions using FP8 precision, SGLang framework, and 2048/2048 token context across six popular LLM architectures.

**Benchmark source**: [LMSYS.org DGX Spark Benchmark Study](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)

---

*This comparison focuses on LLM inference performance. Other workloads (training, multimodal processing, CPU-intensive tasks) may show different performance characteristics due to different bottlenecks and architectural requirements.*
