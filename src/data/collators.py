import torch
import transformers
from typing import Dict, Sequence
from data.utils import IGNORE_INDEX
from transformers import AutoProcessor
from PIL import Image


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        padding_side: str = "right",
        index: str = None,
    ):
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.index = index

    def get_instances_from_key(self, instances: Sequence[Dict], key: str):
        ret_instances = [instance[key] for instance in instances]
        return ret_instances

    def _pad_tokens(self, input_ids, padding_value):
        if self.padding_side == "right":
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=padding_value
            )
        else:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.flip(i, dims=[0]) for i in input_ids],
                batch_first=True,
                padding_value=padding_value,
            ).flip(dims=[1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert isinstance(instances[0], dict)
        return_dct = {}
        if "input_ids" not in instances[0]:
            for key in instances[0].keys():
                key_instances = self.get_instances_from_key(
                    instances=instances, key=key
                )
                return_dct[key] = self(key_instances)
        else:
            input_ids = [instance["input_ids"] for instance in instances]
            input_ids = self._pad_tokens(input_ids, self.tokenizer.pad_token_id)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            return_dct.update({"input_ids": input_ids})
            return_dct.update({"attention_mask": attention_mask})
            if "labels" in instances[0]:
                labels = [instance["labels"] for instance in instances]
                labels = self._pad_tokens(labels, IGNORE_INDEX)
                return_dct.update({"labels": labels})
            if self.index:
                if self.index in instances[0]:
                    return_dct.update(
                        {
                            self.index: torch.tensor(
                                [example[self.index] for example in instances]
                            )
                        }
                    )
                else:
                    raise Warning(f"{self.index} not found in dataset")
        return return_dct


class DataCollatorForMultimodalQADataset(object):
    def __init__(
        self,
        processor_path: str,
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
        system_prompt: str = "You are a helpful assistant.",
        image_max_pixels: int = None,
        padding_side: str = None,
        index: str = None,
        include_text_only: bool = False,
        **kwargs,
    ):
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.tokenizer = self.processor.tokenizer
        if padding_side is not None:
            self.tokenizer.padding_side = padding_side
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.system_prompt = system_prompt
        self.image_max_pixels = image_max_pixels
        self.index = index
        self.include_text_only = include_text_only

    def _resize_image(self, image):
        if self.image_max_pixels is None:
            return image
        width, height = image.size
        pixels = width * height
        if pixels <= self.image_max_pixels:
            return image
        scale = (self.image_max_pixels / pixels) ** 0.5
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(new_size, Image.Resampling.LANCZOS)

    def _processor_images(self, images):
        if self.processor.__class__.__name__ == "Gemma3Processor":
            return [[image] for image in images]
        return images

    def _messages(self, question, answer=None):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        user_content = [{"type": "text", "text": question}]
        messages.append({"role": "user", "content": user_content})
        if answer is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                }
            )
        return messages

    def _messages_with_optional_image(self, question, answer=None, has_image=True):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        user_content = []
        if has_image:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": user_content})
        if answer is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                }
            )
        return messages

    def _pad_and_restore_order(
        self, tensors_by_index, order, padding_value, dtype=None, pad_left=False
    ):
        max_len = max(tensor.shape[0] for tensor in tensors_by_index.values())
        padded = []
        for idx in order:
            tensor = tensors_by_index[idx]
            pad_len = max_len - tensor.shape[0]
            if pad_len:
                pad_shape = (pad_len,) + tensor.shape[1:]
                pad = torch.full(
                    pad_shape,
                    padding_value,
                    dtype=dtype or tensor.dtype,
                    device=tensor.device,
                )
                tensor = torch.cat([pad, tensor], dim=0) if pad_left else torch.cat(
                    [tensor, pad], dim=0
                )
            padded.append(tensor)
        return torch.stack(padded, dim=0)

    def _merge_sub_batches(self, sub_batches, order):
        merged = {}
        for key in ("input_ids", "attention_mask", "labels", "token_type_ids"):
            tensors_by_index = {}
            for indices, batch in sub_batches:
                if key not in batch:
                    continue
                for pos, original_idx in enumerate(indices):
                    tensors_by_index[original_idx] = batch[key][pos]
            if not tensors_by_index:
                continue
            if key == "input_ids":
                padding_value = self.tokenizer.pad_token_id
            elif key in ("attention_mask", "token_type_ids"):
                padding_value = 0
            else:
                padding_value = IGNORE_INDEX
            merged[key] = self._pad_and_restore_order(
                tensors_by_index,
                order,
                padding_value,
                pad_left=self.tokenizer.padding_side == "left",
            )

        for key in sub_batches[0][1].keys():
            if key in merged or key in ("input_ids", "attention_mask", "labels"):
                continue
            values = [batch[key] for _, batch in sub_batches if key in batch]
            if values:
                merged[key] = torch.cat(values, dim=0)
        return merged

    def _make_labels(self, batch, prompt_batch):
        labels = batch["input_ids"].clone()
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1)
        for row_idx, prompt_len in enumerate(prompt_lengths.tolist()):
            if self.tokenizer.padding_side == "left":
                full_len = batch["attention_mask"][row_idx].sum().item()
                prompt_start = max(0, labels.shape[1] - full_len)
                prompt_end = min(prompt_start + prompt_len, labels.shape[1])
                labels[row_idx, prompt_start:prompt_end] = IGNORE_INDEX
            else:
                labels[row_idx, : min(prompt_len, labels.shape[1])] = IGNORE_INDEX
        labels[batch["attention_mask"] == 0] = IGNORE_INDEX
        return labels

    def _collate_qa_subset(self, instances: Sequence[Dict], has_images: bool):
        full_texts = [
            self.processor.apply_chat_template(
                self._messages_with_optional_image(
                    instance["question"], instance["answer"], has_image=has_images
                ),
                tokenize=False,
                add_generation_prompt=False,
            )
            for instance in instances
        ]
        prompt_texts = [
            self.processor.apply_chat_template(
                self._messages_with_optional_image(
                    instance["question"], has_image=has_images
                ),
                tokenize=False,
                add_generation_prompt=True,
            )
            for instance in instances
        ]
        processor_kwargs = {
            "text": full_texts,
            "padding": self.padding,
            "truncation": self.truncation,
            "max_length": self.max_length,
            "return_tensors": "pt",
        }
        prompt_kwargs = dict(processor_kwargs)
        prompt_kwargs["text"] = prompt_texts
        if has_images:
            images = [self._resize_image(instance["image"]) for instance in instances]
            processor_kwargs["images"] = self._processor_images(images)
            prompt_kwargs["images"] = self._processor_images(images)

        batch = self.processor(**processor_kwargs)
        prompt_batch = self.processor(**prompt_kwargs)
        batch["labels"] = self._make_labels(batch, prompt_batch)
        return batch

    def _collate_qa(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        image_indices = [
            idx for idx, instance in enumerate(instances) if instance.get("image") is not None
        ]
        text_indices = [
            idx for idx, instance in enumerate(instances) if instance.get("image") is None
        ]
        sub_batches = []
        if image_indices:
            sub_batches.append(
                (
                    image_indices,
                    self._collate_qa_subset(
                        [instances[idx] for idx in image_indices], has_images=True
                    ),
                )
            )
        if text_indices:
            sub_batches.append(
                (
                    text_indices,
                    self._collate_qa_subset(
                        [instances[idx] for idx in text_indices], has_images=False
                    ),
                )
            )
        batch = self._merge_sub_batches(sub_batches, list(range(len(instances))))

        if self.include_text_only:
            batch["text_only"] = self._collate_qa_subset(
                instances, has_images=False
            )

        if self.index:
            batch[self.index] = torch.tensor(
                [instance[self.index] for instance in instances]
            )
        return batch

    def get_instances_from_key(self, instances: Sequence[Dict], key: str):
        return [instance[key] for instance in instances]

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert isinstance(instances[0], dict)
        if {"image", "question", "answer"}.issubset(instances[0].keys()):
            return self._collate_qa(instances)

        return_dct = {}
        for key in instances[0].keys():
            key_instances = self.get_instances_from_key(instances=instances, key=key)
            return_dct[key] = self(key_instances)
        return return_dct
