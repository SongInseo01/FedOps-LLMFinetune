from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

# 아주 작은 GPT2 구성
config = GPT2Config(
    vocab_size=256,
    n_positions=32,
    n_ctx=32,
    n_embd=64,
    n_layer=1,
    n_head=1,
)

# 모델 생성 및 저장
model = GPT2LMHeadModel(config)
model.save_pretrained("./dummy-basemodel")

# 토크나이저 준비 (최소화 예시)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.save_pretrained("./dummy-basemodel")
