# %% [code]
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
"""
Chunky v1.6.1 — Hard-only Beam N-best + weighted MBR

修复：
- 不再使用 group-beam-search（num_beam_groups / diversity_penalty），避免
  `T5ForConditionalGeneration has no attribute 'transformers-community/group-beam-search'`
- submission 写法改成 id->translation 字典映射，不会再出现全空/格式错

策略：
- 长文本（>CHUNK_THRESHOLD）: clause chunking + 1-best beam
- 短文本：batch 1-best beam
- 极少数 hard 样本（超长 token 或者只按长度触发）：额外用 beam 生成 k 个候选，
  再用 chrF-like MBR + logp 权重从中选一个，覆盖 baseline。
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import re
import logging
import warnings
from pathlib import Path
from typing import List
from dataclasses import dataclass
import random
import math
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )


# %%
@dataclass
class UltraConfig:
    test_data_path: str = (
        "/kaggle/input/deep-past-initiative-machine-translation/test.csv"
    )
    model_path: str = "/kaggle/input/final-byt5/byt5-akkadian-optimized-34x"
    output_dir: str = "/kaggle/working/"

    max_length: int = 512
    batch_size: int = 8
    num_workers: int = 4

    # ---- baseline decode（接近你 35.x 的设置）----
    num_beams: int = 8
    max_new_tokens: int = 512
    length_penalty: float = 1.09
    early_stopping: bool = True
    no_repeat_ngram_size: int = 0

    # ---- hard-only Beam N-best + MBR ----
    use_selective_mbr: bool = True
    mbr_trigger_len: int = 140         # 只对特别长的句子触发
    mbr_trigger_gap: bool = False      # 先关 gap 触发，太常见会乱动
    mbr_max_per_batch: int = 2         # 每 batch 最多处理 2 个 hard 样本

    mbr_k: int = 4                     # 候选数 k (<= num_beams)
    mbr_char_ngram: int = 6            # chrF-like n
    mbr_logp_weight: float = 0.35      # MBR 中 logp 的权重 (0~1)

    # ---- infra ----
    use_mixed_precision: bool = True
    use_better_transformer: bool = True
    use_bucket_batching: bool = True
    use_vectorized_postproc: bool = True
    use_adaptive_beams: bool = True

    aggressive_postprocessing: bool = True
    num_buckets: int = 4

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        if not torch.cuda.is_available():
            self.use_mixed_precision = False
            self.use_better_transformer = False


config = UltraConfig()
print(
    f"\nConfig: beams={config.num_beams}, lp={config.length_penalty}, max_new={config.max_new_tokens}, "
    f"sel_mbr={config.use_selective_mbr}, trig_len={config.mbr_trigger_len}, trig_gap={config.mbr_trigger_gap}, "
    f"k={config.mbr_k}, lam={config.mbr_logp_weight}"
)


# %%
def setup_logging(output_dir: str = "./outputs"):
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


logger = setup_logging(config.output_dir)


# %%
class OptimizedPreprocessor:
    def __init__(self):
        self.patterns = {
            "big_gap": re.compile(r"(\.{3,}|…+|——|……)"),
            "small_gap": re.compile(r"(xx+|\s+x\s+)"),
        }

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        s = pd.Series(texts).fillna("").astype(str)
        s = s.str.replace(self.patterns["big_gap"], "<big_gap>", regex=True)
        s = s.str.replace(self.patterns["small_gap"], "<gap>", regex=True)
        return s.tolist()


preprocessor = OptimizedPreprocessor()


# %%
# Clause-based chunking
CHUNK_MIN_WORDS = 15
CHUNK_MAX_WORDS = 30
CHUNK_THRESHOLD = 50  # Only chunk texts longer than this

CLAUSE_MARKERS = [
    r"KIŠIB\s+",
    r"IGI\s+",
    r"um-ma\s+",
    r"a-na\s+\S+\s+qí-bi",
    r"šu-ma\s+",
    r"\.\s+",
    r"\[\.\.\.\]\s*",
]
CLAUSE_PATTERN = re.compile("|".join(CLAUSE_MARKERS), re.IGNORECASE)


def split_akkadian(
    text: str, max_words: int = CHUNK_MAX_WORDS, min_words: int = CHUNK_MIN_WORDS
) -> List[str]:
    words = text.split()
    if len(words) <= CHUNK_THRESHOLD:
        return [text]

    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        chunk_text = " ".join(current_chunk)
        chunk_len = len(current_chunk)
        is_break = bool(CLAUSE_PATTERN.search(chunk_text + " "))
        if chunk_len >= min_words and is_break:
            chunks.append(chunk_text.strip())
            current_chunk = []
        elif chunk_len >= max_words:
            chunks.append(chunk_text.strip())
            current_chunk = []

    if current_chunk:
        last_chunk = " ".join(current_chunk).strip()
        if last_chunk:
            chunks.append(last_chunk)

    return chunks if chunks else [text]


# %%
class VectorizedPostprocessor:
    def __init__(self, aggressive: bool = True):
        self.aggressive = aggressive
        self.patterns = {
            "gap": re.compile(r"(\[x\]|\(x\)|\bx\b)", re.I),
            "big_gap": re.compile(r"(\.{3,}|…|\[\.+\])"),
            "annotations": re.compile(
                r"\((fem|plur|pl|sing|singular|plural|\?|!)\..\s*\w*\)", re.I
            ),
            "repeated_words": re.compile(r"\b(\w+)(?:\s+\1\b)+"),
            "whitespace": re.compile(r"\s+"),
            "punct_space": re.compile(r"\s+([.,:])"),
            "repeated_punct": re.compile(r"([.,])\1+"),
        }
        self.subscript_trans = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        self.special_chars_trans = str.maketrans("ḫḪ", "hH")
        self.forbidden_chars = '!?()"——<>⌈⌋⌊[]+ʾ/;'
        self.forbidden_trans = str.maketrans("", "", self.forbidden_chars)

    def postprocess_batch(self, translations: List[str]) -> List[str]:
        s = pd.Series(translations)
        valid_mask = s.apply(lambda x: isinstance(x, str) and x.strip())
        if not valid_mask.all():
            s[~valid_mask] = ""

        s = s.str.translate(self.special_chars_trans)
        s = s.str.translate(self.subscript_trans)
        s = s.str.replace(self.patterns["whitespace"], " ", regex=True)
        s = s.str.strip()

        if self.aggressive:
            s = s.str.replace(self.patterns["gap"], "<gap>", regex=True)
            s = s.str.replace(self.patterns["big_gap"], "<big_gap>", regex=True)
            s = s.str.replace("<gap> <gap>", "<big_gap>", regex=False)
            s = s.str.replace("<big_gap> <big_gap>", "<big_gap>", regex=False)
            s = s.str.replace(self.patterns["annotations"], "", regex=True)

            s = s.str.replace("<gap>", "\x00GAP\x00", regex=False)
            s = s.str.replace("<big_gap>", "\x00BIG\x00", regex=False)
            s = s.str.translate(self.forbidden_trans)
            s = s.str.replace("\x00GAP\x00", " <gap> ", regex=False)
            s = s.str.replace("\x00BIG\x00", " <big_gap> ", regex=False)

            s = s.str.replace(r"(\d+)\.5\b", r"\1½", regex=True)
            s = s.str.replace(r"\b0\.5\b", "½", regex=True)
            s = s.str.replace(r"(\d+)\.25\b", r"\1¼", regex=True)
            s = s.str.replace(r"\b0\.25\b", "¼", regex=True)
            s = s.str.replace(r"(\d+)\.75\b", r"\1¾", regex=True)
            s = s.str.replace(r"\b0\.75\b", "¾", regex=True)

            s = s.str.replace(self.patterns["repeated_words"], r"\1", regex=True)
            for n in range(4, 1, -1):
                pattern = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
                s = s.str.replace(pattern, r"\1", regex=True)

            s = s.str.replace(self.patterns["punct_space"], r"\1", regex=True)
            s = s.str.replace(self.patterns["repeated_punct"], r"\1", regex=True)
            s = s.str.replace(self.patterns["whitespace"], " ", regex=True)
            s = s.str.strip().str.strip("-").str.strip()

        return s.tolist()


postprocessor = VectorizedPostprocessor(aggressive=config.aggressive_postprocessing)


# %%
class BucketBatchSampler(Sampler):
    def __init__(
        self, dataset, batch_size: int, num_buckets: int = 4, shuffle: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        lengths = [len(text.split()) for _, text in dataset]
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        bucket_size = max(1, len(sorted_indices) // num_buckets)
        self.buckets = []
        for i in range(num_buckets):
            start = i * bucket_size
            end = None if i == num_buckets - 1 else (i + 1) * bucket_size
            self.buckets.append(sorted_indices[start:end])

        logger.info(f"Created {num_buckets} buckets")

    def __iter__(self):
        for bucket in self.buckets:
            if self.shuffle:
                random.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i : i + self.batch_size]

    def __len__(self):
        return sum(
            (len(b) + self.batch_size - 1) // self.batch_size for b in self.buckets
        )


# %%
class AkkadianDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, preprocessor: OptimizedPreprocessor):
        self.sample_ids = dataframe["id"].tolist()
        raw_texts = dataframe["transliteration"].tolist()
        preprocessed = preprocessor.preprocess_batch(raw_texts)
        self.input_texts = [
            "translate Akkadian to English: " + text for text in preprocessed
        ]
        logger.info(f"Dataset created with {len(self.sample_ids)} samples")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        return self.sample_ids[index], self.input_texts[index]


# %%
class UltraInferenceEngine:
    def __init__(self, config: UltraConfig):
        self.config = config
        self.preprocessor = OptimizedPreprocessor()
        self.postprocessor = VectorizedPostprocessor(
            aggressive=config.aggressive_postprocessing
        )
        self.results: List[tuple[int, str]] = []
        self._load_model()

    def _load_model(self):
        logger.info(f"Loading model from {self.config.model_path}")
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path)
            .to(self.config.device)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded: {num_params:,} parameters")

        if self.config.use_better_transformer and torch.cuda.is_available():
            try:
                from optimum.bettertransformer import BetterTransformer

                logger.info("Applying BetterTransformer...")
                self.model = BetterTransformer.transform(self.model)
                logger.info("BetterTransformer applied")
            except Exception as e:
                logger.warning(f"BetterTransformer skipped/failed: {e}")

    def _collate_fn(self, batch_samples):
        batch_ids = [s[0] for s in batch_samples]
        batch_texts = [s[1] for s in batch_samples]
        tokenized = self.tokenizer(
            batch_texts,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return batch_ids, batch_texts, tokenized

    def _get_adaptive_beam_size(self, input_ids, attention_mask):
        if not self.config.use_adaptive_beams:
            return self.config.num_beams
        lengths = attention_mask.sum(dim=1)
        small = torch.tensor(max(4, self.config.num_beams // 2), device=lengths.device)
        large = torch.tensor(self.config.num_beams, device=lengths.device)
        beam_sizes = torch.where(lengths < 100, small, large)
        return int(beam_sizes[0].item())

    def _is_hard_mask(self, batch_texts: List[str], tokenized) -> List[bool]:
        lengths = tokenized.attention_mask.sum(dim=1).detach().cpu().tolist()
        long_mask = [l >= self.config.mbr_trigger_len for l in lengths]
        if self.config.mbr_trigger_gap:
            gap_mask = [("<gap>" in t) or ("<big_gap>" in t) for t in batch_texts]
        else:
            gap_mask = [False] * len(batch_texts)
        return [a or b for a, b in zip(long_mask, gap_mask)]

    # -------- MBR helpers --------
    def _char_ngrams(self, s: str, n: int):
        s = s.strip()
        if not s:
            return []
        if len(s) < n:
            return [s]
        return [s[i : i + n] for i in range(len(s) - n + 1)]

    def _chrf_like_f1(self, a: str, b: str, n: int) -> float:
        A = self._char_ngrams(a, n)
        B = self._char_ngrams(b, n)
        if not A or not B:
            return 0.0
        ca, cb = Counter(A), Counter(B)
        inter = sum((ca & cb).values())
        prec = inter / max(1, sum(ca.values()))
        rec = inter / max(1, sum(cb.values()))
        if prec + rec == 0:
            return 0.0
        return 2 * prec * rec / (prec + rec)

    def _mbr_select(self, cands: List[str], cand_logps: List[float]) -> str:
        if not cands:
            return ""
        m = max(cand_logps) if cand_logps else 0.0
        ws = [math.exp(lp - m) for lp in cand_logps] if cand_logps else [1.0] * len(cands)
        Z = sum(ws) + 1e-12
        ws = [w / Z for w in ws]

        n = int(self.config.mbr_char_ngram)
        lam = float(self.config.mbr_logp_weight)

        best_i, best_u = 0, -1e18
        for i, yi in enumerate(cands):
            u_cons = 0.0
            for wj, yj in zip(ws, cands):
                u_cons += wj * self._chrf_like_f1(yi, yj, n=n)
            u = (1.0 - lam) * u_cons + lam * math.log(ws[i] + 1e-12)
            if u > best_u:
                best_u, best_i = u, i
        return cands[best_i]

    def _translate_chunks(self, text: str, gen_config: dict) -> str:
        chunks = split_akkadian(text)
        prefix = "translate Akkadian to English: "
        translations = []
        for chunk in chunks:
            inputs = self.tokenizer(
                prefix + chunk,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True,
            ).to(self.config.device)

            if self.config.use_mixed_precision:
                with autocast():
                    outputs = self.model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        **gen_config,
                    )
            else:
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **gen_config,
                )

            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            translations.append(translation.strip())

        return " ".join(translations)

    def run_inference(self, test_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting inference")
        dataset = AkkadianDataset(test_df, self.preprocessor)

        base_gen_config = {
            "max_new_tokens": self.config.max_new_tokens,
            "length_penalty": self.config.length_penalty,
            "early_stopping": self.config.early_stopping,
            "use_cache": True,
            "renormalize_logits": True,
        }
        if self.config.no_repeat_ngram_size > 0:
            base_gen_config["no_repeat_ngram_size"] = self.config.no_repeat_ngram_size

        self.results = []

        # ---------- Phase 1: chunk long texts ----------
        chunked_ids = set()
        with torch.inference_mode():
            for idx in range(len(dataset)):
                sample_id, input_text = dataset[idx]
                raw_text = input_text.replace("translate Akkadian to English: ", "")
                if len(raw_text.split()) > CHUNK_THRESHOLD:
                    gen_config = {**base_gen_config, "num_beams": self.config.num_beams}
                    translation = self._translate_chunks(raw_text, gen_config)
                    cleaned = self.postprocessor.postprocess_batch([translation])[0]
                    self.results.append((sample_id, cleaned))
                    chunked_ids.add(idx)

        if chunked_ids:
            logger.info(f"Chunked {len(chunked_ids)} long texts")

        # ---------- Phase 2: batch translate short texts ----------
        if chunked_ids:
            short_indices = [i for i in range(len(dataset)) if i not in chunked_ids]
            short_dataset = torch.utils.data.Subset(dataset, short_indices)
        else:
            short_dataset = dataset

        if len(short_dataset) == 0:
            return pd.DataFrame(self.results, columns=["id", "translation"])

        if self.config.use_bucket_batching and len(short_dataset) >= self.config.num_buckets:
            batch_sampler = BucketBatchSampler(
                short_dataset, self.config.batch_size, num_buckets=self.config.num_buckets
            )
            dataloader = DataLoader(
                short_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.config.num_workers,
                collate_fn=self._collate_fn,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True if self.config.num_workers > 0 else False,
            )
        else:
            dataloader = DataLoader(
                short_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                collate_fn=self._collate_fn,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True if self.config.num_workers > 0 else False,
            )

        with torch.inference_mode():
            for batch_idx, (batch_ids, batch_texts, tokenized) in enumerate(
                tqdm(dataloader, desc="Translating")
            ):
                try:
                    input_ids = tokenized.input_ids.to(self.config.device)
                    attention_mask = tokenized.attention_mask.to(self.config.device)

                    beam_size = self._get_adaptive_beam_size(input_ids, attention_mask)

                    # 1) baseline 1-best beam
                    gen_config_beam = {**base_gen_config, "num_beams": beam_size}
                    if self.config.use_mixed_precision:
                        with autocast():
                            out_1 = self.model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                **gen_config_beam,
                            )
                    else:
                        out_1 = self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            **gen_config_beam,
                        )
                    translations = self.tokenizer.batch_decode(
                        out_1, skip_special_tokens=True
                    )

                    # 2) hard-only MBR
                    hard_indices: List[int] = []
                    if self.config.use_selective_mbr:
                        hard_mask = self._is_hard_mask(batch_texts, tokenized)
                        hard_indices = [
                            i for i, h in enumerate(hard_mask) if h
                        ][: self.config.mbr_max_per_batch]

                    if hard_indices:
                        k = min(int(self.config.mbr_k), beam_size)
                        gen_config_k = {
                            **base_gen_config,
                            "num_beams": beam_size,
                            "num_return_sequences": k,
                            "return_dict_in_generate": True,
                            "output_scores": True,
                        }

                        hard_input_ids = input_ids[hard_indices]
                        hard_attn = attention_mask[hard_indices]

                        if self.config.use_mixed_precision:
                            with autocast():
                                gen_out = self.model.generate(
                                    input_ids=hard_input_ids,
                                    attention_mask=hard_attn,
                                    **gen_config_k,
                                )
                        else:
                            gen_out = self.model.generate(
                                input_ids=hard_input_ids,
                                attention_mask=hard_attn,
                                **gen_config_k,
                            )

                        seqs = gen_out.sequences  # [hsz*k, L]
                        cand_logps = getattr(gen_out, "sequences_scores", None)
                        if cand_logps is None:
                            cand_logps = torch.zeros(
                                (seqs.size(0),), device=seqs.device
                            )

                        cand_texts = [
                            t.strip()
                            for t in self.tokenizer.batch_decode(
                                seqs, skip_special_tokens=True
                            )
                        ]

                        hsz = len(hard_indices)
                        grouped_texts = [
                            cand_texts[i * k : (i + 1) * k] for i in range(hsz)
                        ]
                        grouped_lp = [
                            cand_logps[i * k : (i + 1) * k]
                            .detach()
                            .cpu()
                            .tolist()
                            for i in range(hsz)
                        ]

                        for loc, idx in enumerate(hard_indices):
                            translations[idx] = self._mbr_select(
                                grouped_texts[loc], grouped_lp[loc]
                            )

                    # postprocess & collect
                    if self.config.use_vectorized_postproc:
                        cleaned = self.postprocessor.postprocess_batch(translations)
                    else:
                        cleaned = [
                            self.postprocessor.postprocess_batch([t])[0]
                            for t in translations
                        ]
                    self.results.extend(zip(batch_ids, cleaned))

                    if torch.cuda.is_available() and batch_idx % 10 == 0:
                        torch.cuda.empty_cache()

                except Exception as e:
                    import traceback

                    logger.error(f"Batch {batch_idx} error: {e}")
                    logger.error(traceback.format_exc())
                    self.results.extend([(bid, "") for bid in batch_ids])
                    continue

        logger.info("Inference completed")
        return pd.DataFrame(self.results, columns=["id", "translation"])


# %%
logger.info(f"Loading test data from {config.test_data_path}")
test_df = pd.read_csv(config.test_data_path, encoding="utf-8")
logger.info(f"Loaded {len(test_df)} test samples")
print(test_df.head())

# %%
engine = UltraInferenceEngine(config)
results_df = engine.run_inference(test_df)

print("\nresults_df shape:", results_df.shape)
print(results_df.head())

# %%
# ---- 安全写 submission：id 映射，不会再全空/格式错 ----
id2t: dict[int, str] = {}
for _id, _t in results_df[["id", "translation"]].values.tolist():
    try:
        iid = int(str(_id).strip())
    except Exception:
        continue
    id2t[iid] = "" if _t is None else str(_t)

sub = pd.DataFrame({"id": test_df["id"].astype(int)})
sub["translation"] = sub["id"].map(id2t).fillna("").astype(str)

output_path = Path(config.output_dir) / "submission.csv"
sub.to_csv(output_path, index=False)

print(f"\nSubmission shape: {sub.shape}")
print(sub.head())
print(
    f"\nSaved submission.csv ({os.path.getsize(output_path):,} bytes) -> {output_path}"
)

assert len(sub) == len(test_df)
assert list(sub.columns) == ["id", "translation"]
assert sub["id"].duplicated().sum() == 0
