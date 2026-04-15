"""Model loading, text generation, and hidden state extraction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

if TYPE_CHECKING:
    from emotion_probes.config import ModelConfig

logger = logging.getLogger(__name__)

# Map string dtype names to torch dtypes
_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _detect_max_memory() -> dict:
    """Auto-detect GPU/CPU memory limits for device_map='auto'.

    Uses total (not available) RAM for CPU to avoid under-allocation
    when memory is temporarily consumed during model loading.
    """
    mem = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory
            # Reserve ~750MB for GPU overhead
            usable = max(0, total - 750 * 1024 * 1024)
            mem[i] = f"{usable // (1024**3)}GiB"
    import psutil
    # Use total RAM (not available) minus a margin for the OS
    cpu_ram = psutil.virtual_memory().total
    cpu_usable = max(0, cpu_ram - 4 * 1024**3)  # reserve 4GB for OS
    mem["cpu"] = f"{cpu_usable // (1024**3)}GiB"
    logger.info("Detected memory: %s", mem)
    return mem


class EmotionProbeModel:
    """Wrapper around a HuggingFace causal LM for probe extraction."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._dtype = _DTYPE_MAP.get(config.torch_dtype, torch.bfloat16)

        logger.info("Loading tokenizer: %s", config.name)
        self.tokenizer = AutoTokenizer.from_pretrained(config.name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-padding is required for correct batch inference with causal LMs
        self.tokenizer.padding_side = "left"

        logger.info("Loading model: %s (dtype=%s, quantize=%s)", config.name, config.torch_dtype, config.quantize)
        load_kwargs: dict = {
            "device_map": config.device_map,
            "torch_dtype": self._dtype,
        }

        if config.quantize == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._dtype,
                bnb_4bit_quant_type="nf4",
            )
        elif config.quantize == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )

        self.model = self._load_model(config.name, load_kwargs, config.quantize)
        self.model.eval()

        self._num_layers: int | None = None
        self._hidden_size: int | None = None
        logger.info("Model loaded successfully.")

    @staticmethod
    def _load_model(name: str, load_kwargs: dict, quantize: str | None):
        """Load model with fallback strategies for auto class and GPU memory."""
        loaders = [AutoModelForCausalLM, AutoModel]
        last_error = None

        for loader in loaders:
            try:
                return loader.from_pretrained(name, **load_kwargs)
            except (ValueError, KeyError) as e:
                last_error = e
                logger.info("%s failed (%s), trying next loader", loader.__name__, e)
                # If quantization fails due to GPU memory, retry with 8-bit + CPU offload
                if "CPU or the disk" in str(e):
                    logger.warning("Model doesn't fit in GPU with current config, falling back to 8-bit + CPU offload")
                    from transformers import BitsAndBytesConfig
                    offload_kwargs = dict(load_kwargs)
                    offload_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True,
                    )
                    offload_kwargs["max_memory"] = _detect_max_memory()
                    offload_kwargs["offload_buffers"] = True
                    try:
                        return loader.from_pretrained(name, **offload_kwargs)
                    except (ValueError, KeyError) as e2:
                        last_error = e2
                        logger.info("8-bit offload also failed: %s", e2)

        raise RuntimeError(f"Could not load model {name}: {last_error}") from last_error

    @property
    def num_layers(self) -> int:
        if self._num_layers is None:
            cfg = self.model.config
            # Some multimodal models nest the text config under .text_config
            if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_hidden_layers"):
                cfg = cfg.text_config
            self._num_layers = cfg.num_hidden_layers
        return self._num_layers

    @property
    def hidden_size(self) -> int:
        if self._hidden_size is None:
            cfg = self.model.config
            if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
                cfg = cfg.text_config
            self._hidden_size = cfg.hidden_size
        return self._hidden_size

    @property
    def target_layer(self) -> int:
        """The target layer (~2/3 through the model), from config or auto-computed."""
        if self.config.target_layer is not None:
            return self.config.target_layer
        return int(self.num_layers * 2 / 3)

    @property
    def device(self) -> torch.device:
        # PreTrainedModel.device handles device_map="auto" correctly
        if hasattr(self.model, "device"):
            return self.model.device
        return next(self.model.parameters()).device

    # ------------------------------------------------------------------
    # Text generation (for story / dialogue creation)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
    ) -> list[str]:
        """Generate text for a batch of prompts using the chat template.

        Each prompt is wrapped as a single user message in the chat template.
        Returns only the generated text (excluding the prompt).
        """
        results: list[str] = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(text, return_tensors="pt", padding=False).to(self.device)
            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                # Build generation kwargs; use the model's own generation config
                # to avoid meta-tensor issues with CPU-offloaded models
                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=64,
                )
                # Avoid pad_token_id == eos_token_id collision that triggers
                # meta-tensor bugs in transformers with CPU-offloaded models
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

                output_ids = self.model.generate(**inputs, **gen_kwargs)
            generated = self.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
            results.append(generated)
        return results

    # ------------------------------------------------------------------
    # Hidden state extraction (for activation probing)
    # ------------------------------------------------------------------

    def get_hidden_states(
        self,
        texts: list[str],
        layers: list[int] | None = None,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Run a forward pass and return hidden states at specified layers.

        Args:
            texts: Raw text strings to process (no chat template applied).
            layers: Layer indices to return. None = [target_layer].

        Returns:
            Tuple of (hidden_states_dict, attention_mask) where:
              - hidden_states_dict maps layer index to tensor (batch, seq_len, hidden_dim)
              - attention_mask is shape (batch, seq_len)
        """
        if layers is None:
            layers = [self.target_layer]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )

        # outputs.hidden_states is a tuple of (num_layers + 1) tensors:
        # index 0 = embedding output, index i = output of layer i
        result = {}
        attention_mask = inputs["attention_mask"]
        for layer_idx in layers:
            if layer_idx < 0 or layer_idx > self.num_layers:
                raise ValueError(f"Layer {layer_idx} out of range [0, {self.num_layers}]")
            hs = outputs.hidden_states[layer_idx].float()  # (batch, seq_len, hidden_dim)
            result[layer_idx] = hs
        return result, attention_mask

    # ------------------------------------------------------------------
    # Unembedding matrix (for logit lens)
    # ------------------------------------------------------------------

    def get_unembedding_matrix(self) -> torch.Tensor:
        """Return the unembedding (lm_head) weight matrix.

        Shape: (vocab_size, hidden_dim)
        Uses get_output_embeddings() for robustness across model architectures
        (handles tied embeddings and nested model wrappers like Gemma 4).
        """
        output_embeddings = self.model.get_output_embeddings()
        if output_embeddings is None:
            raise RuntimeError(
                "Model has no output embeddings (lm_head). "
                "Logit lens requires a model with a language modeling head."
            )
        return output_embeddings.weight.detach().float()
