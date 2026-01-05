# 양자화 기법 구현 분석 문서

이 문서는 Quantized-Reasoning-Models 프로젝트에서 각 양자화 기법이 어떻게 구현되었는지 자세히 분석합니다.

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [Fake Quantization vs Real Quantization](#fake-quantization-vs-real-quantization)
3. [AWQ (Activation-aware Weight Quantization)](#awq-activation-aware-weight-quantization)
4. [GPTQ](#gptq)
5. [KVQuant*](#kvquant)
6. [SmoothQuant](#smoothquant)
7. [FlatQuant](#flatquant)
8. [QuaRot & QuaRot-KV](#quarot--quarot-kv)
9. [공통 유틸리티](#공통-유틸리티)
10. [실행 방법](#실행-방법)

---

## 프로젝트 개요

이 프로젝트는 **"Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models"** 논문의 구현체로, DeepSeek-R1 계열 모델에 대한 다양한 양자화 기법의 효과를 체계적으로 연구합니다.

### 지원하는 양자화 기법

**Fake Quantization (시뮬레이션)**:
- AWQ (W3A16KV16, W4A16KV16)
- GPTQ (W3A16KV16, W4A16KV16)
- KVQuant* (W16A16KV3, W16A16KV4)
- QuaRot-KV (W16A16KV3, W16A16KV4)
- SmoothQuant (W8A8KV8)
- QuaRot (W4A4KV4, W8A8KV8)
- FlatQuant (W4A4KV4, W8A8KV8)

**Real Quantization (실제 양자화)**:
- AWQ (W4A16KV16) - AutoAWQ 사용
- GPTQ (W4A16KV16) - GPTQModel 사용

### 디렉토리 구조

```
Quantized-Reasoning-Models/
├── methods/                    # 각 양자화 기법 구현
│   ├── awq/                   # AWQ 구현
│   ├── quarot_gptq/           # GPTQ & QuaRot 구현
│   ├── kvquant_star/          # KVQuant* 구현
│   ├── smoothquant/           # SmoothQuant 구현
│   ├── flatquant/             # FlatQuant 구현
│   └── utils/                 # 공통 유틸리티
├── scripts/
│   ├── quantization/          # Fake quantization 스크립트
│   └── real_quantization/     # Real quantization 스크립트
├── real_quantization/         # Real quantization 구현
├── vllm_custom/               # 커스텀 vLLM (inference용)
│   ├── model_executor/fake_quantized_models/  # Fake quantized 모델 로더
│   └── third-party/           # 서브모듈 (AutoAWQ, GPTQModel, vLLM 등)
└── lighteval_custom/          # 평가 시스템
```

---

## Fake Quantization vs Real Quantization

### Fake Quantization (가짜 양자화)

**특징**:
- 양자화를 **시뮬레이션**하지만 실제로 메모리는 FP16/BF16 사용
- 양자화 → 역양자화를 즉시 수행하여 정확도 영향만 측정
- 빠른 프로토타이핑과 다양한 설정 실험에 유리
- 메모리/속도 이점 없음

**구현 위치**: `methods/` 디렉토리

**저장 형식**:
- FP16 weight + `fake_quant_config` (양자화 설정)
- 커스텀 모델 클래스 (예: `Qwen2FakeQuantizedForCausalLM`)

**추론 방법**:
- `vllm_custom/model_executor/fake_quantized_models/`의 커스텀 모델 사용
- vLLM에서 로드 시 양자화 시뮬레이션 적용

### Real Quantization (실제 양자화)

**특징**:
- 실제로 INT4 등 저정밀도 포맷으로 변환
- 메모리 사용량 감소 및 추론 속도 향상
- 현재 AWQ와 GPTQ만 지원 (외부 라이브러리 사용)

**구현 위치**: `real_quantization/` 디렉토리

**사용 라이브러리**:
- AWQ: AutoAWQ
- GPTQ: GPTQModel 또는 llm-compressor

**저장 형식**:
- INT4 quantized weights + quantization config
- 외부 라이브러리 호환 포맷

---

## AWQ (Activation-aware Weight Quantization)

### 개요

AWQ는 **activation의 중요도에 따라 weight를 선택적으로 보호**하는 weight-only 양자화 기법입니다.

### 핵심 아이디어

1. **Per-channel Scaling**: Activation 분포를 분석하여 중요한 채널의 weight를 보호
2. **Auto-scaling**: 최적의 스케일 팩터를 자동으로 탐색
3. **Auto-clipping**: Outlier를 클리핑하여 양자화 범위 최적화

### Fake Quantization 구현

**위치**: `methods/awq/`

**주요 파일**:
```
methods/awq/
├── run_awq.py          # 메인 실행 파일
├── pre_quant.py        # AWQ 스케일 탐색 및 적용
├── quantizer.py        # Pseudo-quantization 함수
├── auto_scale.py       # Auto-scaling 구현
├── auto_clip.py        # Auto-clipping 구현
├── qmodule.py          # Quantized 모듈
└── calib_data.py       # Calibration 데이터 로더
```

**실행 흐름** (`run_awq.py:save_awq_model`):

```python
# 1. 모델 로드
model = AutoModelForCausalLM.from_pretrained(args.model)

# 2. AWQ 스케일 탐색 (pre_quant.py:run_awq)
awq_results = run_awq(
    model, tokenizer,
    w_bit=args.w_bits,
    q_config=args.q_config,
    n_samples=128,      # Calibration 샘플 수
    seqlen=512,         # Sequence length
    calib_data="pileval" # Pile 데이터셋 사용
)

# 3. AWQ 스케일 적용 (pre_quant.py:apply_awq)
apply_awq(model_transformers, awq_results)

# 4. Pseudo-quantization (quantizer.py:pseudo_quantize_model_weight)
pseudo_quantize_model_weight(model_transformers, w_bit=args.w_bits, q_config={
    "zero_point": args.w_asym,        # Asymmetric quantization
    "q_group_size": args.w_groupsize, # Group size (128)
})

# 5. 모델 저장
model_transformers.save_pretrained(args.save_qmodel_path)
```

**핵심 함수 분석**:

1. **`run_awq()` (`pre_quant.py`)**:
   ```python
   def run_awq(model, tokenizer, w_bit, q_config, n_samples, seqlen, calib_data):
       # Calibration 데이터로 activation 수집
       # Auto-scaling으로 최적 스케일 탐색
       # Auto-clipping으로 outlier 처리
       return awq_results  # {layer_name: scale_factor}
   ```

2. **`pseudo_quantize_tensor()` (`quantizer.py:64-106`)**:
   ```python
   def pseudo_quantize_tensor(w, n_bit=8, zero_point=True, q_group_size=-1):
       # Group-wise quantization
       if q_group_size > 0:
           w = w.reshape(-1, q_group_size)

       # Asymmetric quantization
       if zero_point:
           max_val = w.amax(dim=1, keepdim=True)
           min_val = w.amin(dim=1, keepdim=True)
           max_int = 2 ** n_bit - 1  # 예: 4bit -> 15
           scales = (max_val - min_val).clamp(min=1e-5) / max_int
           zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)

       # Quantize and dequantize
       w = (torch.clamp(torch.round(w / scales) + zeros, 0, max_int) - zeros) * scales
       return w
   ```

**설정 옵션**:
- `--w_bits`: 3 또는 4 (weight bit-width)
- `--w_groupsize`: 128 (group quantization size)
- `--w_asym`: Asymmetric quantization 사용
- `--n_samples`: 128 (calibration 샘플 수)
- `--seqlen`: 512 (sequence length)
- `--calib_data`: "pileval" (Pile 데이터셋)

### Real Quantization 구현

**위치**: `real_quantization/real_quantization.py` (method="awq-autoawq")

**사용 라이브러리**: AutoAWQ

**실행 흐름**:

```python
from awq import AutoAWQForCausalLM

# 1. AWQ 모델 로드
model = AutoAWQForCausalLM.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model)

# 2. Quantization config 설정
quant_config = {
    "zero_point": True,      # Asymmetric
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"        # GEMM 커널 사용
}

# 3. 양자화 수행
model.quantize(
    tokenizer=tokenizer,
    quant_config=quant_config,
    calib_data="./datasets/pile-val-backup",
    max_calib_samples=128,
    max_calib_seq_len=512,
    duo_scaling=False,       # Duo scaling 비활성화
    apply_clip=True,         # Clipping 적용
)

# 4. 저장
model.save_quantized(args.save_qmodel_path)
```

**차이점**:
- Fake: 양자화 시뮬레이션, FP16 저장, 메모리 이점 없음
- Real: 실제 INT4 변환, 메모리 ~4배 감소, GEMM 커널 가속

---

## GPTQ

### 개요

GPTQ는 **Layer-wise optimal quantization**을 수행하는 weight-only 양자화 기법으로, Hessian 정보를 활용하여 각 레이어의 quantization error를 최소화합니다.

### 핵심 아이디어

1. **Hessian-based Quantization**: Second-order 정보(Hessian)를 사용하여 최적 양자화
2. **Layer-wise Reconstruction**: 각 레이어별로 순차적으로 양자화
3. **Activation Order**: Activation의 중요도 순으로 양자화 (optional)
4. **Group Quantization**: Group 단위로 양자화하여 정확도 향상

### Fake Quantization 구현

**위치**: `methods/quarot_gptq/`

**주요 파일**:
```
methods/quarot_gptq/
├── save_fake_quant.py     # 메인 실행 파일
└── rotation_utils.py      # QuaRot용 회전 변환 (GPTQ에는 미사용)
```

**공통 유틸리티** (`methods/utils/`):
```
methods/utils/
├── gptq_utils.py          # GPTQ 알고리즘 구현
├── quant_utils.py         # Quantization 유틸리티
├── model_utils.py         # 모델 로더
├── data_utils.py          # 데이터 로더
└── hadamard_utils.py      # Hadamard 변환 (QuaRot용)
```

**실행 흐름** (`save_fake_quant.py:main`):

```python
# 1. 모델 로드
model = model_utils.get_model(args.model, args.seqlen)

# 2. (Optional) QuaRot 회전 적용
if args.rotate:
    rotation_utils.fuse_layer_norms(model)
    rotation_utils.rotate_model(model, args)

# 3. Activation quantization wrapper 추가
quant_utils.add_actquant(model)

# 4. Weight quantization
if args.w_bits < 16:
    if not args.w_rtn:  # GPTQ 사용
        # Calibration 데이터 로드
        trainloader = data_utils.get_loaders(
            args.cal_dataset,      # "reasoning-numina-math-1.5"
            nsamples=args.nsamples,
            seqlen=model.seqlen
        )
        # GPTQ 수행
        quantizers = gptq_utils.gptq_fwrd(model, trainloader, DEV, args)
    else:  # RTN (Round-To-Nearest) 사용
        quantizers = gptq_utils.rtn_fwrd(model, DEV, args)

# 5. 모델 저장
model.save_pretrained(args.save_qmodel_path)
model.config.fake_quant_config = {
    "tp": args.tp,
    "w_bits": args.w_bits,
    "w_clip": args.w_clip,
    "a_bits": args.a_bits,
    ...
}
model.config.save_pretrained(args.save_qmodel_path)
```

**핵심 함수 분석**:

1. **`gptq_fwrd()` (`methods/utils/gptq_utils.py`)**:
   ```python
   def gptq_fwrd(model, dataloader, dev, args):
       # Layer-by-layer GPTQ quantization
       layers = model.model.layers  # Transformer layers

       for layer in layers:
           # 1. Collect input-output pairs
           layer = layer.to(dev)
           handles = []
           for name, module in layer.named_modules():
               if isinstance(module, nn.Linear):
                   # Hook to collect activations
                   handles.append(module.register_forward_hook(forward_hook))

           # 2. Forward calibration data
           for batch in dataloader:
               layer(batch)

           # 3. GPTQ quantization for each linear layer
           for name, module in layer.named_modules():
               if isinstance(module, nn.Linear):
                   quantizer = GPTQ(module)
                   quantizer.quantize(
                       W=module.weight,
                       H=hessian_info,     # Collected Hessian
                       bits=args.w_bits,
                       groupsize=args.w_groupsize,
                       actorder=args.act_order  # Activation ordering
                   )
   ```

2. **GPTQ 알고리즘 (`GPTQ.quantize()`)**:
   ```python
   # Optimal Brain Quantization (OBQ) 알고리즘 변형
   for i in range(n_columns):
       # 1. Hessian의 역행렬을 이용한 error 계산
       Q = torch.linalg.cholesky(H_inv)

       # 2. Weight를 양자화하고 error 계산
       w_quant = quantize(W[:, i])
       error = (W[:, i] - w_quant) / H_inv[i, i]

       # 3. Error를 나머지 weight에 전파
       W[:, i+1:] -= error.unsqueeze(1) @ Q[i, i+1:].unsqueeze(0)

       # 4. Hessian 업데이트
       H_inv = update_hessian(H_inv, i)
   ```

**특징**:
- **Reasoning Calibration**: NuminaMath-1.5 데이터셋 사용 (수학 문제 reasoning 데이터)
- **Activation Ordering**: `--act_order` 옵션으로 활성화
- **Group Size**: 128 (group-wise quantization)
- **Clipping**: `--w_clip` 옵션으로 outlier 처리

**설정 옵션** (`scripts/quantization/gptq.sh`):
```bash
python -m methods.quarot_gptq.save_fake_quant \
    --model ${model} \
    --w_bits 4 \
    --w_clip \           # Weight clipping
    --w_asym \           # Asymmetric quantization
    --w_groupsize 128 \
    --act_order \        # Activation ordering
    --tp ${tp} \         # Tensor parallelism
    --cal_dataset reasoning-numina-math-1.5  # Reasoning calibration data
```

### Real Quantization 구현

**위치**: `real_quantization/real_quantization.py` (method="gptq-gptqmodel")

**사용 라이브러리**: GPTQModel

**실행 흐름**:

```python
from gptqmodel import GPTQModel, QuantizeConfig

# 1. Calibration 데이터 준비 (Reasoning 데이터)
tokenizer = AutoTokenizer.from_pretrained(args.model)
ds = get_reasoning_calib_dataset(
    model_name=args.model_name,
    tokenizer=tokenizer,
    n_samples=128,
    seqlen=2048,
    return_attention_mask=True
)

# 2. Quantization config
quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    sym=False,           # Asymmetric
    desc_act=True,       # Activation ordering
    static_groups=True,
    mse=2.4,            # MSE-based clipping threshold
)

# 3. 모델 로드 및 양자화
model = GPTQModel.load(args.model, quant_config)
model.quantize(calibration_dataset=ds, batch_size=2)

# 4. 저장
model.save(args.save_qmodel_path)
```

**주요 차이점**:
- **Calibration 데이터**: Reasoning 데이터셋 (NuminaMath-1.5 생성 데이터) 사용
- **MSE Clipping**: `mse=2.4` 파라미터로 outlier 처리
- **Desc Act**: Activation ordering 활성화

---

## KVQuant*

### 개요

KVQuant*는 **KV cache를 양자화**하는 기법으로, 특히 **Pre-RoPE K cache의 bias를 보정**하여 정확도를 향상시킵니다.

### 핵심 아이디어

1. **KV Cache Quantization**: Weight는 FP16 유지, KV cache만 양자화 (W16A16KV3/4)
2. **Pre-RoPE K Quantization**: RoPE 적용 전의 K를 양자화하여 정확도 향상
3. **Bias Correction**: Pre-RoPE K의 per-channel bias를 보정
4. **Group-wise V Quantization**: V cache는 group-wise로 양자화

### 구현 위치

**위치**: `methods/kvquant_star/`

**주요 파일**:
```
methods/kvquant_star/
├── save_fake_quant.py   # 메인 실행 파일
└── rope_utils.py        # RoPE 관련 유틸리티
```

### 실행 흐름

**스크립트**: `scripts/quantization/kvquant_star.sh`

```bash
# 1.5B/7B 모델: k_pre_bias 사용
python -m methods.kvquant_star.save_fake_quant \
    --model ${model} \
    --k_bits 4 --k_asym --k_pre_bias \    # Pre-RoPE K bias 보정
    --v_bits 4 --v_asym --v_groupsize 128 \
    --seqlen 512 --nsamples 512 \
    --cal_dataset pileval \
    --tp ${tp}

# 14B 이상: k_pre_bias 미사용
python -m methods.kvquant_star.save_fake_quant \
    --model ${model} \
    --k_bits 4 --k_asym \                 # Bias 보정 없음
    --v_bits 4 --v_asym --v_groupsize 128 \
    --seqlen 512 --nsamples 512 \
    --cal_dataset pileval \
    --tp ${tp}
```

**코드 분석** (`save_fake_quant.py:main`):

```python
# 1. 모델 로드
model = model_utils.get_model(args.model, args.seqlen)
tokenizer = AutoTokenizer.from_pretrained(args.model)

# 2. Pre-RoPE K 통계치 수집
pre_rope_k_scale_zero = rope_utils.get_pre_rope_k_stats(
    args, model, tokenizer,
    cal_dataset=args.cal_dataset,
    num_samples=args.nsamples,  # 512
    seq_len=args.seqlen         # 512
)
# 반환값: {layer_idx: {"scale": [dim, 1], "zero": [dim, 1]}}

# 3. 모델 저장
model.save_pretrained(args.save_qmodel_path)

# 4. Pre-RoPE K 통계치 저장
torch.save(pre_rope_k_scale_zero,
           f"{args.save_qmodel_path}/pre_rope_k_scale_zero.pth")

# 5. Config 저장
model.config.architectures = ["Qwen2KVQuantStarForCausalLM"]
model.config.fake_quant_config = {
    "tp": args.tp,
    "k_bits": args.k_bits,
    "k_asym": args.k_asym,
    "k_scale_path": f"{args.save_qmodel_path}/pre_rope_k_scale_zero.pth",
    "k_pre_bias": args.k_pre_bias,  # 1.5B/7B만 True
    "v_bits": args.v_bits,
    "v_asym": args.v_asym,
    "v_groupsize": args.v_groupsize,
}
```

**핵심 함수**: `get_pre_rope_k_stats()` (`rope_utils.py`)

```python
def get_pre_rope_k_stats(args, model, tokenizer, cal_dataset, num_samples, seq_len):
    """
    Calibration 데이터로 Pre-RoPE K의 min/max 수집
    """
    pre_rope_k_stats = {}

    # 1. Hook 등록: K projection 후, RoPE 전에 activation 수집
    for layer_idx, layer in enumerate(model.model.layers):
        def forward_hook(module, input, output):
            # Q, K, V projection 후의 값
            # K를 RoPE 적용 전에 수집
            k_states = output[1]  # Key states

            # Per-channel min/max 계산
            if layer_idx not in pre_rope_k_stats:
                pre_rope_k_stats[layer_idx] = {
                    "min": k_states.min(dim=0, keepdim=True)[0],
                    "max": k_states.max(dim=0, keepdim=True)[0]
                }
            else:
                pre_rope_k_stats[layer_idx]["min"] = torch.min(
                    pre_rope_k_stats[layer_idx]["min"],
                    k_states.min(dim=0, keepdim=True)[0]
                )
                pre_rope_k_stats[layer_idx]["max"] = torch.max(
                    pre_rope_k_stats[layer_idx]["max"],
                    k_states.max(dim=0, keepdim=True)[0]
                )

        layer.self_attn.register_forward_hook(forward_hook)

    # 2. Calibration 데이터 forward
    dataloader = get_calibration_data(cal_dataset, num_samples, seq_len)
    for batch in dataloader:
        model(**batch)

    # 3. Scale/Zero 계산
    for layer_idx in pre_rope_k_stats:
        min_val = pre_rope_k_stats[layer_idx]["min"]
        max_val = pre_rope_k_stats[layer_idx]["max"]

        # Asymmetric quantization
        n_bit = args.k_bits
        max_int = 2 ** n_bit - 1
        scale = (max_val - min_val) / max_int
        zero = -torch.round(min_val / scale).clamp(0, max_int)

        pre_rope_k_stats[layer_idx] = {
            "scale": scale,
            "zero": zero
        }

    return pre_rope_k_stats
```

### Inference 시 동작

**위치**: `vllm_custom/model_executor/fake_quantized_models/qwen2_fake_quantized.py` 또는 `llama_fake_quantized.py`

```python
class Qwen2KVQuantStarForCausalLM:
    def __init__(self, config):
        # Pre-RoPE K scale/zero 로드
        k_scale_path = config.fake_quant_config["k_scale_path"]
        self.pre_rope_k_scale_zero = torch.load(k_scale_path)

    def forward(self, ...):
        # Attention에서 K cache 양자화
        for layer_idx, layer in enumerate(self.model.layers):
            # 1. K projection
            k_states = layer.self_attn.k_proj(hidden_states)

            # 2. Pre-RoPE K 양자화 (k_pre_bias 사용 시)
            if self.config.fake_quant_config["k_pre_bias"]:
                scale = self.pre_rope_k_scale_zero[layer_idx]["scale"]
                zero = self.pre_rope_k_scale_zero[layer_idx]["zero"]

                # Quantize
                k_quant = torch.clamp(
                    torch.round(k_states / scale) + zero,
                    0, 2**k_bits - 1
                )
                # Dequantize
                k_states = (k_quant - zero) * scale

            # 3. RoPE 적용
            k_states = apply_rotary_pos_emb(k_states, position_ids)

            # 4. V cache 양자화 (group-wise)
            v_states = layer.self_attn.v_proj(hidden_states)
            v_states = fake_quantize_tensor(
                v_states,
                n_bit=v_bits,
                group_size=v_groupsize,
                asym=True
            )
```

### 설정 옵션

- `--k_bits`: 3 또는 4 (K cache bit-width)
- `--k_asym`: Asymmetric quantization
- `--k_pre_bias`: Pre-RoPE K bias 보정 (1.5B/7B만)
- `--v_bits`: 3 또는 4 (V cache bit-width)
- `--v_asym`: Asymmetric quantization
- `--v_groupsize`: 128 (V cache group size)
- `--seqlen`: 512
- `--nsamples`: 512
- `--cal_dataset`: "pileval"

---

## SmoothQuant

### 개요

SmoothQuant는 **Activation과 Weight의 난이도를 균형있게 조정**하여 W8A8 양자화를 가능하게 하는 기법입니다.

### 핵심 아이디어

1. **Channel-wise Smoothing**: Activation의 outlier를 weight로 이동
   - `Y = (X * s^(-1)) * (W * s)` 변환 (s는 smoothing factor)
2. **Per-token Dynamic Quantization**: Activation은 per-token으로 동적 양자화
3. **W8A8KV8**: Weight, Activation, KV cache 모두 INT8로 양자화

### 구현 위치

**위치**: `methods/smoothquant/`

**주요 파일**:
```
methods/smoothquant/
├── save_fake_quant.py        # 메인 실행 파일
└── smoothquant_utils.py      # SmoothQuant 유틸리티
```

### 실행 흐름

**스크립트**: `scripts/quantization/smoothquant.sh`

```bash
CUDA_VISIBLE_DEVICES=${device} \
python -m methods.smoothquant.save_fake_quant \
    --model ${model} \
    --w_bits 8 --w_clip \                # Weight INT8 + clipping
    --a_bits 8 --a_asym \                # Activation INT8 (asymmetric)
    --k_bits 8 --k_asym --k_groupsize 128 \   # K cache INT8 (group-wise)
    --v_bits 8 --v_asym --v_groupsize 128 \   # V cache INT8 (group-wise)
    --seqlen 2048 --nsamples 128 \
    --cal_dataset reasoning-numina-math-1.5 \  # Reasoning calibration
    --tp ${tp}
```

**코드 분석** (`save_fake_quant.py:main`):

```python
# 1. 모델 로드
model = model_utils.get_model(args.model, args.seqlen)

# 2. SmoothQuant scales 계산
act_scales_path = f"{args.save_qmodel_path}/act_scales.pt"
if not os.path.exists(act_scales_path):
    # Calibration 데이터로 activation 수집
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    act_scales = smoothquant_utils.get_act_scales(
        args, model, tokenizer,
        cal_dataset=args.cal_dataset,  # "reasoning-numina-math-1.5"
        num_samples=args.nsamples,     # 128
        seq_len=args.seqlen            # 2048
    )
    torch.save(act_scales, act_scales_path)
else:
    act_scales = torch.load(act_scales_path)

# 3. Smoothing 적용 (weight에 scale 병합)
smoothquant_utils.smooth_lm(model, act_scales, alpha=args.smooth_alpha)

# 4. Weight quantization (GPTQ 또는 RTN)
if args.w_bits < 16:
    trainloader = data_utils.get_loaders(args.cal_dataset, ...)
    quantizers = gptq_utils.gptq_fwrd(model, trainloader, DEV, args)

# 5. 모델 저장
model.save_pretrained(args.save_qmodel_path)
model.config.architectures = ["Qwen2FakeQuantizedForCausalLM"]
model.config.fake_quant_config = {
    "w_bits": 8, "a_bits": 8,
    "k_bits": 8, "v_bits": 8,
    ...
}
```

**핵심 함수 분석**:

1. **`get_act_scales()` (`smoothquant_utils.py`)**:
   ```python
   def get_act_scales(args, model, tokenizer, cal_dataset, num_samples, seq_len):
       """
       Calibration 데이터로 activation의 max 값 수집
       """
       act_scales = {}

       # 1. Hook 등록: Linear layer의 입력 activation 수집
       for name, module in model.named_modules():
           if isinstance(module, nn.Linear):
               def forward_hook(module, input, output):
                   # Per-channel max 계산
                   x = input[0]  # [batch, seq, hidden]
                   max_val = x.abs().amax(dim=(0, 1))  # [hidden]

                   if name not in act_scales:
                       act_scales[name] = max_val
                   else:
                       act_scales[name] = torch.max(act_scales[name], max_val)

               module.register_forward_hook(forward_hook)

       # 2. Calibration 데이터 forward
       dataloader = get_calibration_data(cal_dataset, num_samples, seq_len)
       for batch in dataloader:
           model(**batch)

       return act_scales
   ```

2. **`smooth_lm()` (`smoothquant_utils.py`)**:
   ```python
   def smooth_lm(model, act_scales, alpha=0.5):
       """
       Smoothing factor를 weight에 병합

       Smoothing factor: s = max(|X|)^alpha / max(|W|)^(1-alpha)
       """
       for name, module in model.named_modules():
           if isinstance(module, nn.Linear):
               # 1. Smoothing factor 계산
               x_max = act_scales[name]           # Activation max
               w_max = module.weight.abs().amax(dim=0)  # Weight max (per output channel)

               s = (x_max.pow(alpha) / w_max.pow(1 - alpha))  # Smoothing factor

               # 2. Weight에 smoothing 적용: W' = W * s
               module.weight.data *= s.unsqueeze(0)

               # 3. 다음 LayerNorm의 weight에 역수 적용: gamma' = gamma / s
               #    (LayerNorm이 있는 경우, X' = X / s를 보상)
               next_ln = find_next_layernorm(module)
               if next_ln is not None:
                   next_ln.weight.data /= s

       # Note: Inference 시 activation을 s로 나누는 대신,
       #       weight를 s로 곱하고 LayerNorm을 s로 나눠서 등가 변환
   ```

### Inference 시 동작

SmoothQuant는 **pre-processing이 완료된 weight를 사용**하므로, inference 시에는 일반적인 INT8 양자화만 수행합니다.

```python
class Qwen2FakeQuantizedForCausalLM:
    def forward(self, ...):
        # Weight는 이미 smoothing 적용됨
        for layer in self.model.layers:
            # Activation quantization (per-token dynamic)
            x = fake_quantize_activation(
                hidden_states,
                n_bit=8,
                asym=True,
                per_token=True  # Per-token dynamic quantization
            )

            # Weight quantization (static, per-channel)
            w = fake_quantize_weight(
                layer.self_attn.q_proj.weight,
                n_bit=8,
                per_channel=True
            )

            # Linear operation
            output = x @ w.T
```

**Per-token Dynamic Quantization**:
```python
def fake_quantize_activation(x, n_bit=8, asym=True, per_token=True):
    """
    x: [batch, seq, hidden]
    """
    if per_token:
        # Per-token quantization (각 토큰마다 scale 계산)
        max_val = x.abs().amax(dim=-1, keepdim=True)  # [batch, seq, 1]
        min_val = x.amin(dim=-1, keepdim=True)

    # Asymmetric quantization
    max_int = 2 ** n_bit - 1
    scale = (max_val - min_val) / max_int
    zero = -torch.round(min_val / scale).clamp(0, max_int)

    # Quantize & dequantize
    x_quant = torch.clamp(torch.round(x / scale) + zero, 0, max_int)
    return (x_quant - zero) * scale
```

### 설정 옵션

- `--w_bits`: 8 (Weight INT8)
- `--w_clip`: Weight clipping 사용
- `--a_bits`: 8 (Activation INT8)
- `--a_asym`: Asymmetric activation quantization
- `--k_bits`, `--v_bits`: 8 (KV cache INT8)
- `--k_groupsize`, `--v_groupsize`: 128
- `--smooth_alpha`: 0.5 (기본값, smoothing factor α)
- `--seqlen`: 2048
- `--nsamples`: 128
- `--cal_dataset`: "reasoning-numina-math-1.5"

---

## FlatQuant

### 개요

FlatQuant는 **Learnable transformations**을 활용하여 activation의 분포를 평탄화(flatten)하고, W4A4KV4 또는 W8A8KV8 양자화를 달성하는 기법입니다.

### 핵심 아이디어

1. **Learnable Transformations**: 각 레이어에 학습 가능한 변환 행렬 추가
   - `Y = X @ Q @ R @ W` (Q, R: learnable matrices)
2. **Flattening Outliers**: Outlier를 flatten하여 양자화 친화적인 분포 생성
3. **Calibration-based Training**: Calibration 데이터로 변환 행렬 학습
4. **LWC (Learnable Weight Clipping)**: Weight clipping threshold 학습
5. **LAC (Learnable Activation Clipping)**: Activation clipping threshold 학습

### 구현 위치

**위치**: `methods/flatquant/`

**주요 파일**:
```
methods/flatquant/
├── main.py              # 메인 실행 파일
├── gptq_utils.py        # GPTQ 유틸리티 (weight quantization)
└── flatquant/           # FlatQuant 서브모듈 (submodule)
    ├── utils.py
    ├── args_utils.py
    ├── model_utils.py
    ├── train_utils.py   # Calibration 학습
    ├── flat_utils.py    # Transformation 관리
    └── eval_utils.py
```

### 실행 흐름

**스크립트**: `scripts/quantization/flatquant.sh`

```bash
python -m methods.flatquant.main \
    --model ${model} \
    --w_bits 4 --a_bits 4 --a_asym \
    --k_bits 4 --k_asym --k_groupsize 128 \
    --v_bits 4 --v_asym --v_groupsize 128 \
    --cali_bsz 4 --epoch 15 --flat_lr 5e-3 \    # 학습 설정
    --lwc --lac --cali_trans --add_diag \        # FlatQuant 옵션
    --cali_dataset wikitext2 --seqlen 4096 \     # Calibration 데이터
    --output_dir ./outputs/modelzoo/flatquant/logs \
    --save_matrix \                               # Transformation 저장
    --deactive_amp --direct_inv \
    --tp ${tp} \
    --exp_name w4a4kv4tp${tp} \
    --save_qmodel_path ./outputs/modelzoo/flatquant/${model_name}-flatquant-w4a4kv4-tp${tp}
```

**코드 분석** (`main.py:main`):

```python
# 1. 모델 로드
model, apply_flatquant_to_model = model_utils.get_model(args.model, args.seqlen)

# 2. Calibration 데이터 로드
trainloader = data_utils.get_loaders(
    args.cali_dataset,    # "wikitext2"
    nsamples=args.nsamples,
    seqlen=model.seqlen
)

# 3. FlatQuant 적용
if args.quantize:
    # 3-1. FlatQuant 레이어 추가
    model = apply_flatquant_to_model(args, model)

    # 3-2. Transformation 학습
    if args.cali_trans or args.add_diag or args.lwc or args.lac:
        train_utils.cali_flat_quant(
            args, model, trainloader, DEV,
            logger=logger,
            start_layer_idx=0
        )

    # 3-3. Transformation 저장
    if args.save_matrix:
        flat_utils.save_flat_matrices(args, model)

    # 3-4. Reparameterize (변환 행렬을 weight에 병합)
    flat_utils.reparameterize_model(model)

# 4. Weight quantization (GPTQ)
if args.w_bits < 16:
    if args.gptq:
        quantizers = gptq_utils.gptq_fwrd(model, trainloader, DEV, args)
    else:
        quantizers = gptq_utils.rtn_fwrd(model, DEV, args)

# 5. 모델 저장
model.save_pretrained(args.save_qmodel_path)

# 6. Transformation 파라미터 저장
flat_parameters = torch.load(os.path.join(args.exp_dir, "flat_matrices.pth"))
torch.save(flat_parameters, f"{args.save_qmodel_path}/flat_matrices.pth")

model.config.architectures = ["Qwen2FlatQuantForCausalLM"]
model.config.fake_quant_config = {
    "w_bits": 4, "a_bits": 4,
    "k_bits": 4, "v_bits": 4,
    "lwc": args.lwc,
    "lac": args.lac,
    "direct_inv": args.direct_inv,
    ...
}
```

**핵심 함수 분석**:

1. **`apply_flatquant_to_model()` (`model_utils.py`)**:
   ```python
   def apply_flatquant_to_model(args, model):
       """
       각 Linear layer에 FlatQuant transformation 추가
       """
       for name, module in model.named_modules():
           if isinstance(module, nn.Linear):
               # Learnable transformation 추가
               flatquant_module = FlatQuantLinear(
                   module,
                   lwc=args.lwc,  # Learnable weight clipping
                   lac=args.lac,  # Learnable activation clipping
                   cali_trans=args.cali_trans,  # Calibration transformation
                   add_diag=args.add_diag       # Diagonal matrix 추가
               )
               set_module_by_name(model, name, flatquant_module)

       return model
   ```

2. **`FlatQuantLinear` 모듈**:
   ```python
   class FlatQuantLinear(nn.Module):
       def __init__(self, linear, lwc=True, lac=True, cali_trans=True, add_diag=True):
           super().__init__()
           self.linear = linear

           # Learnable transformations
           if cali_trans:
               # Q: Orthogonal transformation (학습 가능)
               self.Q = nn.Parameter(torch.eye(linear.in_features))
               # R: Rotation matrix (학습 가능)
               self.R = nn.Parameter(torch.eye(linear.in_features))

           if add_diag:
               # D: Diagonal scaling (학습 가능)
               self.diag = nn.Parameter(torch.ones(linear.in_features))

           if lwc:
               # Weight clipping threshold (학습 가능)
               self.w_clip_factor = nn.Parameter(torch.tensor(1.0))

           if lac:
               # Activation clipping threshold (학습 가능)
               self.a_clip_factor = nn.Parameter(torch.tensor(1.0))

       def forward(self, x):
           # 1. Activation transformation: X' = X @ Q @ R @ D
           if self.cali_trans:
               x = x @ self.Q @ self.R
           if self.add_diag:
               x = x * self.diag

           # 2. Activation clipping & quantization
           if self.lac:
               clip_val = x.abs().amax() * self.a_clip_factor
               x = x.clamp(-clip_val, clip_val)
           x = fake_quantize_activation(x, n_bit=self.a_bits)

           # 3. Weight clipping & quantization
           w = self.linear.weight
           if self.lwc:
               clip_val = w.abs().amax() * self.w_clip_factor
               w = w.clamp(-clip_val, clip_val)
           w = fake_quantize_weight(w, n_bit=self.w_bits)

           # 4. Linear operation
           return F.linear(x, w, self.linear.bias)
   ```

3. **`cali_flat_quant()` (`train_utils.py`)**:
   ```python
   def cali_flat_quant(args, model, trainloader, dev, logger, start_layer_idx=0):
       """
       Calibration 데이터로 transformation 학습
       """
       layers = model.model.layers

       for layer_idx in range(start_layer_idx, len(layers)):
           layer = layers[layer_idx].to(dev)

           # 각 layer의 learnable parameters 최적화
           optimizer = torch.optim.Adam([
               {"params": [m.Q for m in layer.modules() if hasattr(m, 'Q')]},
               {"params": [m.R for m in layer.modules() if hasattr(m, 'R')]},
               {"params": [m.diag for m in layer.modules() if hasattr(m, 'diag')]},
               {"params": [m.w_clip_factor for m in layer.modules() if hasattr(m, 'w_clip_factor')]},
               {"params": [m.a_clip_factor for m in layer.modules() if hasattr(m, 'a_clip_factor')]},
           ], lr=args.flat_lr)

           # Calibration 학습 (여러 epoch)
           for epoch in range(args.epoch):
               for batch in trainloader:
                   # Forward
                   output = layer(batch)

                   # Loss: Reconstruction error
                   # (양자화 전후 출력 차이 최소화)
                   with torch.no_grad():
                       target = layer_without_quant(batch)
                   loss = F.mse_loss(output, target)

                   # Backward
                   optimizer.zero_grad()
                   loss.backward()
                   optimizer.step()

                   # Orthogonality constraint (Q, R)
                   # Q, R을 orthogonal하게 유지
                   with torch.no_grad():
                       for m in layer.modules():
                           if hasattr(m, 'Q'):
                               m.Q.data = orthogonalize(m.Q.data)
                           if hasattr(m, 'R'):
                               m.R.data = orthogonalize(m.R.data)
   ```

4. **`reparameterize_model()` (`flat_utils.py`)**:
   ```python
   def reparameterize_model(model):
       """
       학습된 transformation을 weight에 병합

       W' = (Q @ R @ D)^(-1) @ W
       """
       for name, module in model.named_modules():
           if isinstance(module, FlatQuantLinear):
               # Transformation 합성
               trans = torch.eye(module.linear.in_features)
               if module.cali_trans:
                   trans = trans @ module.Q @ module.R
               if module.add_diag:
                   trans = trans @ torch.diag(module.diag)

               # Weight에 역변환 적용
               # W' = trans^(-1) @ W
               trans_inv = torch.inverse(trans)
               module.linear.weight.data = module.linear.weight.data @ trans_inv.T

               # Transformation 제거 (이제 weight에 포함됨)
               del module.Q, module.R, module.diag
   ```

### Inference 시 동작

Reparameterization 후에는 transformation이 weight에 병합되어 있으므로, inference 시에는 일반적인 양자화만 수행합니다.

```python
class Qwen2FlatQuantForCausalLM:
    def __init__(self, config):
        # FlatQuant matrices 로드
        flat_matrices_path = f"{model_path}/flat_matrices.pth"
        self.flat_params = torch.load(flat_matrices_path)

    def forward(self, ...):
        # Weight는 이미 reparameterize됨
        for layer in self.model.layers:
            # Activation quantization (learnable clipping 적용)
            x = fake_quantize_activation(
                hidden_states,
                n_bit=4,
                clip_factor=self.flat_params[layer_idx]["a_clip_factor"]
            )

            # Weight quantization (learnable clipping 적용)
            w = fake_quantize_weight(
                layer.self_attn.q_proj.weight,
                n_bit=4,
                clip_factor=self.flat_params[layer_idx]["w_clip_factor"]
            )
```

### 설정 옵션

**FlatQuant 옵션**:
- `--lwc`: Learnable weight clipping 활성화
- `--lac`: Learnable activation clipping 활성화
- `--cali_trans`: Calibration transformation (Q, R) 활성화
- `--add_diag`: Diagonal scaling (D) 활성화
- `--direct_inv`: Direct inversion 방식 사용

**학습 옵션**:
- `--cali_bsz`: 4 (Calibration batch size)
- `--epoch`: 15 (학습 epoch 수)
- `--flat_lr`: 5e-3 (Learning rate)
- `--cali_dataset`: "wikitext2"
- `--seqlen`: 4096 (1.5B/7B), 2048 (14B+)

**양자화 옵션**:
- `--w_bits`, `--a_bits`, `--k_bits`, `--v_bits`: 4 또는 8
- `--k_groupsize`, `--v_groupsize`: 128
- `--gptq`: GPTQ weight quantization 사용

---

## QuaRot & QuaRot-KV

### 개요

QuaRot는 **Hadamard rotation**을 활용하여 outlier를 분산시키고 양자화를 용이하게 하는 기법입니다.

- **QuaRot**: Weight, Activation, KV cache 모두 양자화 (W4A4KV4, W8A8KV8)
- **QuaRot-KV**: KV cache만 양자화 (W16A16KV3, W16A16KV4)

### 핵심 아이디어

1. **Hadamard Rotation**: 직교 변환으로 outlier 분산
   - `Y = X @ H @ W` (H: Hadamard matrix)
2. **Layer Norm Fusion**: LayerNorm을 weight에 병합
3. **Online Hadamard Transform**: Inference 시 효율적인 Hadamard 변환

### 구현 위치

**위치**: `methods/quarot_gptq/` (GPTQ와 동일 디렉토리)

**주요 파일**:
```
methods/quarot_gptq/
├── save_fake_quant.py     # 메인 실행 파일 (GPTQ와 공유)
└── rotation_utils.py      # Hadamard rotation 유틸리티
```

### 실행 흐름

**QuaRot**: `scripts/quantization/quarot.sh`

```bash
python -m methods.quarot_gptq.save_fake_quant \
    --model ${model} \
    --rotate \                    # Rotation 활성화
    --w_bits 4 --w_rtn \         # RTN weight quantization
    --a_bits 4 --a_asym \
    --k_bits 4 --k_asym --k_groupsize 128 \
    --v_bits 4 --v_asym --v_groupsize 128 \
    --tp ${tp}
```

**QuaRot-KV**: `scripts/quantization/quarot_kv.sh`

```bash
python -m methods.quarot_gptq.save_fake_quant \
    --model ${model} \
    --rotate \                    # Rotation 활성화
    --w_bits 16 \                # Weight 양자화 안함
    --k_bits 4 --k_asym --k_groupsize 128 \
    --v_bits 4 --v_asym --v_groupsize 128 \
    --tp ${tp}
```

**코드 분석** (`save_fake_quant.py:main`):

```python
# 1. 모델 로드
model = model_utils.get_model(args.model, args.seqlen)

# 2. Rotation 적용
if args.rotate:
    # 2-1. LayerNorm fusion
    rotation_utils.fuse_layer_norms(model)

    # 2-2. Hadamard rotation
    rotation_utils.rotate_model(model, args)

    # 2-3. Activation quantization wrapper 추가
    quant_utils.add_actquant(model)

    # 2-4. Online Hadamard transform 설정
    qlayers = quant_utils.find_qlayers(model)
    for name in qlayers:
        if 'down_proj' in name:
            # MLP down_proj: Full Hadamard
            had_dim = model.config.intermediate_size // args.tp
            had_K, K = hadamard_utils.get_hadK(had_dim)
            qlayers[name].had_dim = had_dim
            qlayers[name].online_full_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K

        if 'o_proj' in name and args.tp == 1:
            # Attention o_proj: Head-wise Hadamard
            had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
            qlayers[name].online_partial_had = True
            qlayers[name].had_K = had_K
            qlayers[name].K = K
            qlayers[name].had_dim = model.config.hidden_size // model.config.num_attention_heads

# 3. Weight quantization (RTN 또는 GPTQ)
if args.w_bits < 16:
    if args.w_rtn:
        quantizers = gptq_utils.rtn_fwrd(model, DEV, args)
    else:
        quantizers = gptq_utils.gptq_fwrd(model, trainloader, DEV, args)

# 4. 모델 저장
if args.rotate:
    model.config.architectures = ["Qwen2QuaRotForCausalLM"]
elif args.k_bits < 16 or args.v_bits < 16:
    model.config.architectures = ["Qwen2QuaRotKVForCausalLM"]
```

**핵심 함수 분석**:

1. **`fuse_layer_norms()` (`rotation_utils.py`)**:
   ```python
   def fuse_layer_norms(model):
       """
       LayerNorm을 다음 Linear layer의 weight에 병합

       Y = LayerNorm(X) @ W
         = (X - mean) / std * gamma @ W
         = X @ (W * gamma / std) - (mean / std * gamma) @ W

       병합 후:
       W' = W * gamma / std
       bias' = bias - (mean / std * gamma) @ W
       """
       for layer in model.model.layers:
           # Self-attention input LayerNorm
           ln = layer.input_layernorm
           gamma = ln.weight

           # Q, K, V projection에 병합
           for proj in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
               proj.weight.data *= gamma.unsqueeze(0)

           # LayerNorm 제거 (identity로 교체)
           layer.input_layernorm = nn.Identity()
   ```

2. **`rotate_model()` (`rotation_utils.py`)**:
   ```python
   def rotate_model(model, args):
       """
       Hadamard rotation 적용

       Y = X @ H @ W

       구현: W' = H @ W (weight를 미리 회전)
       Inference 시: X @ W' = X @ (H @ W)
       """
       for layer in model.model.layers:
           # Hadamard matrix 생성
           H = hadamard_utils.get_hadamard_matrix(model.config.hidden_size)

           # Q, K, V, O projection 회전
           for proj in [layer.self_attn.q_proj, layer.self_attn.k_proj,
                       layer.self_attn.v_proj, layer.self_attn.o_proj]:
               # W' = H^T @ W^T -> W'^T = W @ H
               proj.weight.data = proj.weight.data @ H

           # MLP projection 회전
           layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data @ H
           layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data @ H

           # Down projection에는 역회전
           H_inv = H.T  # Hadamard matrix는 직교 행렬
           layer.mlp.down_proj.weight.data = H_inv @ layer.mlp.down_proj.weight.data
   ```

3. **Online Hadamard Transform** (`hadamard_utils.py`):
   ```python
   def get_hadK(n):
       """
       Hadamard matrix를 K개의 작은 행렬로 분해

       H_n = H_{n/K} ⊗ H_K (Kronecker product)

       이를 이용하여 O(n log n) -> O(K * n) 복잡도로 감소
       """
       # K 선택 (일반적으로 K = log2(n))
       K = int(np.log2(n))
       had_K = hadamard_matrix(K)

       return had_K, K

   def online_hadamard_transform(x, had_K, K, had_dim):
       """
       Online Hadamard transform (효율적 구현)

       x: [batch, seq, n]
       """
       n = x.shape[-1]

       # Reshape: [batch, seq, n/K, K]
       x = x.reshape(*x.shape[:-1], n // K, K)

       # Apply K-way Hadamard: [batch, seq, n/K, K] @ [K, K]
       x = x @ had_K

       # Reshape back: [batch, seq, n]
       x = x.reshape(*x.shape[:-2], n)

       return x
   ```

### Inference 시 동작

```python
class Qwen2QuaRotForCausalLM:
    def forward(self, ...):
        for layer in self.model.layers:
            # 1. Activation은 이미 회전된 weight와 곱해짐
            # (Rotation은 weight에 미리 적용됨)

            # 2. MLP down_proj: Online Hadamard transform
            if hasattr(layer.mlp.down_proj, 'online_full_had'):
                x = layer.mlp.gate_proj(hidden_states)
                x = layer.mlp.act_fn(x)

                # Online Hadamard (activation에 적용)
                x = online_hadamard_transform(
                    x,
                    had_K=layer.mlp.down_proj.had_K,
                    K=layer.mlp.down_proj.K,
                    had_dim=layer.mlp.down_proj.had_dim
                )

                # Quantization
                x = fake_quantize_activation(x, n_bit=4)

                output = layer.mlp.down_proj(x)
```

### 설정 옵션

**QuaRot**:
- `--rotate`: Hadamard rotation 활성화
- `--w_bits`: 4 또는 8
- `--w_rtn`: RTN weight quantization (GPTQ 대신)
- `--a_bits`, `--k_bits`, `--v_bits`: 4 또는 8

**QuaRot-KV**:
- `--rotate`: Hadamard rotation 활성화
- `--w_bits`: 16 (weight 양자화 안함)
- `--k_bits`, `--v_bits`: 3 또는 4

---

## 공통 유틸리티

모든 양자화 기법이 공유하는 유틸리티 코드들입니다.

**위치**: `methods/utils/`

### 주요 파일

1. **`model_utils.py`**: 모델 로더
   ```python
   def get_model(model_path, seqlen, hf_token=None):
       """HuggingFace 모델 로드"""
       model = AutoModelForCausalLM.from_pretrained(
           model_path,
           torch_dtype="auto",
           device_map="cpu"
       )
       model.seqlen = seqlen
       return model
   ```

2. **`data_utils.py`**: Calibration 데이터 로더
   ```python
   def get_loaders(dataset_name, nsamples, seed, model, seqlen):
       """
       Calibration 데이터셋 로드

       지원 데이터셋:
       - "pileval": Pile 데이터셋 (일반 텍스트)
       - "wikitext2": WikiText-2 (일반 텍스트)
       - "reasoning-numina-math-1.5": NuminaMath-1.5 생성 데이터 (수학 reasoning)
       """
       if dataset_name == "pileval":
           dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
       elif dataset_name == "wikitext2":
           dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
       elif dataset_name == "reasoning-numina-math-1.5":
           # 미리 생성된 reasoning 데이터 로드
           dataset = load_generated_reasoning_data(model_name)

       # Tokenize
       tokenizer = AutoTokenizer.from_pretrained(model)
       dataloader = []
       for _ in range(nsamples):
           sample = random.choice(dataset)
           tokens = tokenizer(sample["text"], return_tensors="pt",
                            max_length=seqlen, truncation=True)
           dataloader.append(tokens)

       return dataloader
   ```

3. **`quant_utils.py`**: 양자화 유틸리티
   ```python
   def add_actquant(model):
       """Activation quantization wrapper 추가"""
       for name, module in model.named_modules():
           if isinstance(module, nn.Linear):
               module = ActQuantLinear(module)

   def find_qlayers(model):
       """Quantization이 적용될 layer 검색"""
       qlayers = {}
       for name, module in model.named_modules():
           if isinstance(module, (ActQuantLinear, nn.Linear)):
               qlayers[name] = module
       return qlayers
   ```

4. **`gptq_utils.py`**: GPTQ 알고리즘 (여러 기법에서 공유)
   - `gptq_fwrd()`: GPTQ forward (layer-wise quantization)
   - `rtn_fwrd()`: RTN (Round-To-Nearest) quantization

5. **`hadamard_utils.py`**: Hadamard 변환 (QuaRot용)
   - `get_hadamard_matrix()`: Hadamard matrix 생성
   - `get_hadK()`: Online Hadamard transform용 분해
   - `online_hadamard_transform()`: 효율적 Hadamard 변환

### Fake Quantization 공통 함수

모든 fake quantization에서 사용하는 핵심 함수:

```python
def fake_quantize_tensor(w, n_bit=8, zero_point=True, q_group_size=-1):
    """
    Pseudo-quantization (simulate quantize-dequantize)

    Args:
        w: Weight or activation tensor
        n_bit: Bit-width (3, 4, 8, etc.)
        zero_point: Asymmetric quantization 사용 여부
        q_group_size: Group quantization size (-1이면 per-channel)

    Returns:
        Dequantized tensor (same dtype as input)
    """
    org_shape = w.shape

    # Group quantization
    if q_group_size > 0:
        w = w.reshape(-1, q_group_size)

    # Asymmetric quantization
    if zero_point:
        max_val = w.amax(dim=-1, keepdim=True)
        min_val = w.amin(dim=-1, keepdim=True)
        max_int = 2 ** n_bit - 1
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp(0, max_int)
    else:
        # Symmetric quantization
        max_val = w.abs().amax(dim=-1, keepdim=True)
        max_int = 2 ** (n_bit - 1) - 1
        scales = max_val / max_int
        zeros = 0

    # Quantize
    w_int = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)

    # Dequantize
    w_dequant = (w_int - zeros) * scales

    return w_dequant.reshape(org_shape)
```

---

## 실행 방법

### 1. Fake Quantization

**개별 양자화 기법 실행**:

```bash
# AWQ
bash scripts/quantization/awq.sh \
    ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B \
    4 \    # Tensor Parallelism
    0      # GPU device

# GPTQ
bash scripts/quantization/gptq.sh \
    ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B \
    4 \
    0

# KVQuant*
bash scripts/quantization/kvquant_star.sh \
    ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B \
    4 \
    0

# SmoothQuant
bash scripts/quantization/smoothquant.sh \
    ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B \
    4 \
    0,1,2,3  # Multi-GPU

# FlatQuant
bash scripts/quantization/flatquant.sh \
    ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B \
    4 \
    0

# QuaRot
bash scripts/quantization/quarot.sh \
    ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B \
    4 \
    0

# QuaRot-KV
bash scripts/quantization/quarot_kv.sh \
    ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B \
    4 \
    0
```

**출력 디렉토리**:
```
outputs/modelzoo/
├── awq/
│   ├── DeepSeek-R1-Distill-Qwen-7B-awq-w3g128-tp4/
│   └── DeepSeek-R1-Distill-Qwen-7B-awq-w4g128-tp4/
├── gptq/
│   ├── DeepSeek-R1-Distill-Qwen-7B-gptq-w3g128-tp4/
│   └── DeepSeek-R1-Distill-Qwen-7B-gptq-w4g128-tp4/
├── kvquant_star/
│   ├── DeepSeek-R1-Distill-Qwen-7B-kvquant_star-kv3-tp4/
│   └── DeepSeek-R1-Distill-Qwen-7B-kvquant_star-kv4-tp4/
├── smoothquant/
│   └── DeepSeek-R1-Distill-Qwen-7B-smoothquant-w8a8kv8-tp4/
├── flatquant/
│   ├── DeepSeek-R1-Distill-Qwen-7B-flatquant-w4a4kv4-tp4/
│   └── DeepSeek-R1-Distill-Qwen-7B-flatquant-w8a8kv8-tp4/
└── ...
```

### 2. Real Quantization

```bash
# AWQ (AutoAWQ)
bash scripts/real_quantization/awq.sh \
    ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B \
    0

# GPTQ (GPTQModel)
bash scripts/real_quantization/gptq.sh \
    ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B \
    0
```

**출력 디렉토리**:
```
outputs/modelzoo/real_quantization/
├── awq-autoawq/
│   └── DeepSeek-R1-Distill-Qwen-7B-quantized.awq-autoawq-w4g128/
└── gptq-gptqmodel/
    └── DeepSeek-R1-Distill-Qwen-7B-quantized.gptq-gptqmodel-w4g128/
```

### 3. Inference

**개별 모델 추론**:

```bash
# Base model (FP16)
bash scripts/inference/inference.sh \
    ./modelzoo/DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B \
    0,1,2,3

# Fake quantized model
bash scripts/inference/inference.sh \
    ./outputs/modelzoo/gptq/DeepSeek-R1-Distill-Qwen-7B-gptq-w4g128-tp4 \
    0,1,2,3

# Real quantized model
bash scripts/inference/inference.sh \
    ./outputs/modelzoo/real_quantization/gptq-gptqmodel/DeepSeek-R1-Distill-Qwen-7B-quantized.gptq-gptqmodel-w4g128 \
    0,1,2,3
```

**모든 양자화 모델 추론** (수정된 스크립트):

```bash
bash scripts/analysis/run_all_quantized_models.sh 0 42
```

이 스크립트는:
- MATH-500: 100개 샘플
- AIME-90: 45개 샘플
- 총 9개 모델 × 2개 데이터셋 = 18개 실험

**출력 디렉토리**:
```
outputs/inference/
├── DeepSeek-R1-Distill-Qwen-7B/
│   ├── custom_math_500_seed_42.jsonl
│   └── custom_aime_90_seed_42.jsonl
├── DeepSeek-R1-Distill-Qwen-7B-gptq-w4g128-tp4/
│   ├── custom_math_500_seed_42.jsonl
│   └── custom_aime_90_seed_42.jsonl
└── ...
```

### 4. 결과 분석

```bash
# 정확도 테이블 출력
python -m make_stats_table --stats acc

# Response length 테이블 출력
python -m make_stats_table --stats length

# Real quantization 결과 포함
python -m make_stats_table --stats acc \
    --methods quantized.gptq-gptqmodel-w4g128 quantized.awq-autoawq-w4g128
```

---

## 요약 비교표

| 기법 | 양자화 대상 | Bit-width | Calibration 데이터 | 특징 |
|------|-------------|-----------|-------------------|------|
| **AWQ** | Weight only | W3/W4 | Pile (일반 텍스트) | Activation-aware scaling |
| **GPTQ** | Weight only | W3/W4 | NuminaMath-1.5 (reasoning) | Hessian-based optimization |
| **KVQuant*** | KV cache only | KV3/KV4 | Pile | Pre-RoPE K bias correction |
| **SmoothQuant** | W+A+KV | W8A8KV8 | NuminaMath-1.5 | Activation smoothing |
| **FlatQuant** | W+A+KV | W4A4KV4, W8A8KV8 | WikiText-2 | Learnable transformations |
| **QuaRot** | W+A+KV | W4A4KV4, W8A8KV8 | - | Hadamard rotation |
| **QuaRot-KV** | KV cache only | KV3/KV4 | - | Hadamard rotation (KV only) |

---

## 참고 문헌

이 프로젝트는 다음 연구들을 기반으로 합니다:

1. **AWQ**: Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"
2. **GPTQ**: Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
3. **KVQuant**: Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization"
4. **SmoothQuant**: Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
5. **FlatQuant**: Liu et al., "FlatQuant: Flattening Outliers with Better Quantization for Large Language Models"
6. **QuaRot**: Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs"

---

**문서 작성일**: 2025-01-05
**프로젝트 버전**: COLM 2025 Accepted Version
