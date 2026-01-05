# Comprehensive Analysis of Quantized Reasoning Models

**Analysis Date**: 2026-01-05 22:20:14
**Random Seed**: 42
**Total Models Analyzed**: 6
**Total Datasets**: 2
**Total Responses**: 3,540

---

## Executive Summary

### AIME-90

- **Baseline Accuracy**: 25.56%
- **Best Quantized Model**: DeepSeek-R1-Distill-Qwen-1.5B-awq-w4g128-tp1
  - Accuracy: 20.00% (-5.56pp)
- **Worst Quantized Model**: DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1
  - Accuracy: 6.67% (-18.89pp)

### MATH-500

- **Baseline Accuracy**: 85.00%
- **Best Quantized Model**: DeepSeek-R1-Distill-Qwen-1.5B-kvquant_star-kv4-tp1
  - Accuracy: 84.20% (-0.80pp)
- **Worst Quantized Model**: DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1
  - Accuracy: 52.60% (-32.40pp)


---

## AIME-90 Analysis

### Performance Comparison

| Model | Accuracy | Avg Words | Wait Count | Token Diversity |
|-------|----------|-----------|------------|------------------|
| DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1 | 6.67% | 17312 | 1531.4 | 0.0552 |
| DeepSeek-R1-Distill-Qwen-1.5B-awq-w4g128-tp1 | 20.00% | 9913 | 169.3 | 0.1288 |
| DeepSeek-R1-Distill-Qwen-1.5B-gptq-w3g128-tp1 | 10.00% | 9183 | 97.0 | 0.1180 |
| DeepSeek-R1-Distill-Qwen-1.5B-gptq-w4g128-tp1 | 18.89% | 8309 | 111.4 | 0.1459 |
| DeepSeek-R1-Distill-Qwen-1.5B-kvquant_star-kv4-tp1 | 20.00% | 9637 | 123.3 | 0.1470 |
| DeepSeek-R1-Distill-Qwen-1.5B | 25.56% | 8711 | 107.8 | 0.1548 |


---

## MATH-500 Analysis

### Performance Comparison

| Model | Accuracy | Avg Words | Wait Count | Token Diversity |
|-------|----------|-----------|------------|------------------|
| DeepSeek-R1-Distill-Qwen-1.5B-awq-w3g128-tp1 | 52.60% | 11327 | 968.7 | 0.1185 |
| DeepSeek-R1-Distill-Qwen-1.5B-awq-w4g128-tp1 | 83.40% | 2959 | 46.7 | 0.2427 |
| DeepSeek-R1-Distill-Qwen-1.5B-gptq-w3g128-tp1 | 71.40% | 2935 | 41.5 | 0.2366 |
| DeepSeek-R1-Distill-Qwen-1.5B-gptq-w4g128-tp1 | 83.00% | 2976 | 36.4 | 0.2493 |
| DeepSeek-R1-Distill-Qwen-1.5B-kvquant_star-kv4-tp1 | 84.20% | 3025 | 39.5 | 0.2530 |
| DeepSeek-R1-Distill-Qwen-1.5B | 85.00% | 2983 | 33.2 | 0.2505 |


---

## Quantization Method Comparison

### AWQ

- **Number of Models**: 2
- **Average Accuracy**: 40.67%
- **Average Response Length**: 10378 words
- **Average Token Diversity**: 0.1363

### Baseline

- **Number of Models**: 1
- **Average Accuracy**: 55.28%
- **Average Response Length**: 5847 words
- **Average Token Diversity**: 0.2027

### GPTQ

- **Number of Models**: 2
- **Average Accuracy**: 45.82%
- **Average Response Length**: 5851 words
- **Average Token Diversity**: 0.1874

### KV-Quant*

- **Number of Models**: 1
- **Average Accuracy**: 52.10%
- **Average Response Length**: 6331 words
- **Average Token Diversity**: 0.2000


---

## Error Pattern Analysis

**Total Errors Analyzed**: 1151

### Error Categories

- **Severe Repetition (Wait > 1000)**: 141 (12.3%)
- **Moderate Repetition (100 < Wait ≤ 1000)**: 579 (50.3%)
- **Garbled Text**: 160 (13.9%)
- **Missing Answer**: 438 (38.1%)
- **Low Token Diversity (< 0.1)**: 588 (51.1%)


---

## Key Findings and Recommendations

### Key Findings

**AIME-90**:
- Baseline accuracy: 25.56%
- 0 quantized models maintain <5pp accuracy drop

**MATH-500**:
- Baseline accuracy: 85.00%
- 3 quantized models maintain <5pp accuracy drop
- Best quantized model: DeepSeek-R1-Distill-Qwen-1.5B-kvquant_star-kv4-tp1 (84.20%)

### Recommendations

1. **For high-accuracy requirements**: Use baseline FP16 or 4-bit quantization (AWQ/GPTQ)
2. **For memory-constrained scenarios**: 4-bit quantization offers best accuracy-efficiency trade-off
3. **Avoid 3-bit quantization** for reasoning tasks due to severe repetition degeneration
4. **KV-cache quantization** shows promising results with minimal accuracy loss


---

## Data Availability

All intermediate data, tables, and statistics are available in:
- Intermediate data: `./analysis_results/intermediate_data/`
- Tables (CSV, LaTeX, Markdown): `./analysis_results/tables/`
- Statistics: `./analysis_results/statistics/`

