import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from datasets import load_dataset
from tqdm import tqdm
import wandb

wandb.init(project="hyper-language", name="Train HL BPE Tokenizer v1")

# Params
vocab_size = 128000
max_samples = 1000000  # 1M+ for large vocab
hl_vocab_size = 2000  # Larger base HL

print("Step 1: Generate HL corpus from C4 multi-lang")
from hl_tokenizer import HLTokenizer  # Use current simple HL
hl_tok = HLTokenizer(vocab_size=hl_vocab_size)

dataset_configs = [
    ('allenai/c4', 'en', 'text'), ('allenai/c4', 'zh', 'text'), ('allenai/c4', 'ja', 'text'),
    ('allenai/c4', 'fr', 'text'), ('allenai/c4', 'de', 'text'), ('allenai/c4', 'es', 'text'),
    ('allenai/c4', 'it', 'text'), ('allenai/c4', 'ko', 'text'), ('allenai/c4', 'ru', 'text')
]

hl_corpus = []
samples_per_lang = max_samples // len(dataset_configs)
for dataset_name, subdataset, field in dataset_configs:
    print(f"Loading {subdataset}...")
    ds = load_dataset(dataset_name, subdataset, split='train', streaming=True)
    count = 0
    for ex in ds:
        text = ex[field].strip()
        if len(text) > 50:
            hl_text = hl_tok.encode(text)
            hl_corpus.append(hl_text)
            count += 1
        if count >= samples_per_lang:
            break

print(f"Generated {len(hl_corpus)} HL texts")

# Save corpus
os.makedirs('hl_corpus', exist_ok=True)
with open('hl_corpus/hl_texts.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(hl_corpus))

print("Step 2: Train BPE tokenizer on HL corpus")
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
    min_frequency=5,  # Higher for large corpus
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

files = ['hl_corpus/hl_texts.txt']
tokenizer.train(files, trainer)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.save('hl_tokenizer.json')

wandb.save('hl_tokenizer.json')
wandb.finish()

print("Done! Use from tokenizers import Tokenizer; tok = Tokenizer.from_file('hl_tokenizer.json')")
print("Next: integrate into train_baseline_hl.py for comparison")