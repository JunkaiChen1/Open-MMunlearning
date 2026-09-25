import torch
import json
import ast
import random
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset

from data.utils import load_hf_dataset, preprocess_chat_instance, add_dataset_index


class QADataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        few_shot_dataset_hf_args=None,
        max_length=512,
        predict_with_generate=False,
    ):
        super(QADataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        self.fs_data = None
        if few_shot_dataset_hf_args is not None:
            raw_data = load_hf_dataset(**few_shot_dataset_hf_args)
            self.fs_data = {}
            self.fs_data[question_key] = raw_data[question_key]
            self.fs_data[answer_key] = raw_data[answer_key]
        self.template_args = template_args
        self.question_key = question_key
        self.answer_key = answer_key
        self.predict_with_generate = predict_with_generate

    def __len__(self):
        return len(self.data)

    def _process_sample(self, question, answer, index=-1):
        if self.fs_data is None:
            prompt_msgs, response_msgs = [question], [answer]
        else:
            prompt_msgs = self.fs_data[self.question_key] + [question]
            response_msgs = self.fs_data[self.answer_key] + [answer]
        tokenized_data = preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            prompt_msgs,
            response_msgs,
            self.max_length,
            self.predict_with_generate,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
            "index": index,
        }
        return item_dct

    def __getitem__(self, idx):
        question = self.data[idx][self.question_key]
        answer = self.data[idx][self.answer_key]
        index = self.data[idx]["index"]
        if isinstance(answer, str):
            item = self._process_sample(question=question, answer=answer, index=index)
        elif isinstance(answer, list):
            item = {}
            for i, ans in enumerate(answer):
                sample_item = self._process_sample(
                    question=question, answer=ans, index=index
                )
                item[i] = sample_item
        else:
            raise NotImplementedError("answer format not found")
        return item


class QAwithIdkDataset(QADataset):
    def __init__(self, idk_path, return_original=True, *args, **kwargs):
        self.idk_path = idk_path
        self.return_original = return_original
        self.idk_responses = open(self.idk_path, "r").readlines()
        super().__init__(*args, **kwargs)

    def item_with_idk(self, question):
        rand_pos = torch.randint(0, len(self.idk_responses), (1,)).item()
        idk_response = self.idk_responses[rand_pos].strip()
        idk_item = self._process_sample(question=question, answer=idk_response)
        return idk_item

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        if isinstance(item, dict):
            return_item = {"original": item}
            idk_item = self.item_with_idk(question)
            return_item["alternate"] = idk_item
            # return_item = [item, idk_item]
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                idk_item = self.item_with_idk(question)
                return_item["alternate"] = idk_item
                # return_item.append([sample_item, idk_item])
        return return_item if self.return_original else return_item["alternate"]


class ImageQADataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        image_key="image",
        id_key="ID",
        qa_pairs_key=None,
        qa_question_key="Question",
        qa_answer_key="Answer",
        metadata_sources=None,
        idk_path=None,
        max_length=512,
        tokenize=True,
        predict_with_generate=False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.template_args = template_args
        self.question_key = question_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.id_key = id_key
        self.qa_pairs_key = qa_pairs_key
        self.qa_question_key = qa_question_key
        self.qa_answer_key = qa_answer_key
        self.metadata_sources = set(metadata_sources) if metadata_sources else None
        self.idk_responses = None
        if idk_path is not None:
            with open(idk_path, "r", encoding="utf-8") as idk_file:
                self.idk_responses = [line.strip() for line in idk_file if line.strip()]
            if not self.idk_responses:
                raise ValueError(f"No IDK responses found in {idk_path}")
        self.max_length = max_length
        self.tokenize = tokenize
        self.predict_with_generate = predict_with_generate
        self.data = load_hf_dataset(**hf_args)
        self.samples = self._build_samples()

    def __len__(self):
        return len(self.samples)

    def _parse_qa_pairs(self, value):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = ast.literal_eval(value)
        return value

    def _load_image(self, image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, dict):
            if image.get("bytes") is not None:
                return Image.open(BytesIO(image["bytes"])).convert("RGB")
            if image.get("path") is not None:
                return Image.open(image["path"]).convert("RGB")
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        raise ValueError(f"Unsupported image value: {type(image)}")

    def _build_samples(self):
        samples = []
        for row_idx, row in enumerate(self.data):
            sample_id = row.get(self.id_key, row_idx) if self.id_key else row_idx
            image = row.get(self.image_key, None) if self.image_key else None
            if self.qa_pairs_key:
                qa_pairs = self._parse_qa_pairs(row[self.qa_pairs_key])
                for qa_idx, qa_pair in enumerate(qa_pairs):
                    source = qa_pair.get("source") or qa_pair.get("Source")
                    if self.metadata_sources and source not in self.metadata_sources:
                        continue
                    question = qa_pair.get(self.qa_question_key, "")
                    answer = qa_pair.get(self.qa_answer_key, "")
                    if question and answer:
                        sample_image = None if qa_pair.get("has_image") is False else image
                        samples.append(
                            {
                                "question": question,
                                "answer": answer,
                                "index": len(samples),
                                "id": sample_id,
                                "qa_index": qa_idx,
                                "image": sample_image,
                            }
                        )
            else:
                samples.append(
                    {
                        "question": row[self.question_key],
                        "answer": row[self.answer_key],
                        "index": row_idx,
                        "id": sample_id,
                        "image": image,
                    }
                )
        return samples

    def _process_sample(self, question, answer, index=-1):
        tokenized_data = preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            [question],
            [answer],
            self.max_length,
            self.predict_with_generate,
        )
        return {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
            "index": index,
        }

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if not self.tokenize:
            item = {
                "image": self._load_image(sample["image"])
                if sample["image"] is not None
                else None,
                "question": sample["question"],
                "answer": sample["answer"],
                "index": sample["index"],
                "id": sample.get("id"),
                "qa_index": sample.get("qa_index"),
            }
        else:
            item = self._process_sample(
                question=sample["question"],
                answer=sample["answer"],
                index=sample["index"],
            )
            if "id" in sample:
                item["id"] = sample["id"]
            if "qa_index" in sample:
                item["qa_index"] = sample["qa_index"]
        if self.idk_responses is None:
            return item

        idk_answer = random.choice(self.idk_responses)
        if not self.tokenize:
            alternate = dict(item)
            alternate["answer"] = idk_answer
        else:
            alternate = self._process_sample(
                question=sample["question"],
                answer=idk_answer,
                index=sample["index"],
            )
            if "id" in sample:
                alternate["id"] = sample["id"]
            if "qa_index" in sample:
                alternate["qa_index"] = sample["qa_index"]
        return {"original": item, "alternate": alternate}


class UMUBenchQADataset(ImageQADataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        qa_key="MM_QA",
        qa_keys_source=None,
        image_key="image",
        id_key="ID",
        qa_keys=None,
        idk_path=None,
        max_length=512,
        tokenize=True,
        predict_with_generate=False,
    ):
        self.qa_keys_source = qa_keys_source or [qa_key]
        self.qa_keys = qa_keys
        self.idk_responses = None
        if idk_path is not None:
            with open(idk_path, "r", encoding="utf-8") as idk_file:
                self.idk_responses = [line.strip() for line in idk_file if line.strip()]
            if not self.idk_responses:
                raise ValueError(f"No IDK responses found in {idk_path}")
        super().__init__(
            hf_args=hf_args,
            template_args=template_args,
            tokenizer=tokenizer,
            image_key=image_key,
            id_key=id_key,
            max_length=max_length,
            tokenize=tokenize,
            predict_with_generate=predict_with_generate,
        )

    def _parse_umu_qa(self, value):
        if isinstance(value, str):
            return ast.literal_eval(value)
        return value

    def _build_samples(self):
        samples = []
        for row_idx, row in enumerate(self.data):
            sample_id = row.get(self.id_key, row_idx) if self.id_key else row_idx
            image = row.get(self.image_key, None) if self.image_key else None
            for qa_source in self.qa_keys_source:
                sample_image = image if qa_source == "MM_QA" else None
                qa_pairs = self._parse_umu_qa(row[qa_source])
                questions = qa_pairs["question"]
                answers = qa_pairs["answer"]
                qa_keys = self.qa_keys or questions.keys()
                for qa_name in qa_keys:
                    question = questions.get(qa_name, "")
                    answer = answers.get(qa_name, "")
                    if question and answer:
                        samples.append(
                            {
                                "question": question,
                                "answer": answer,
                                "index": len(samples),
                                "id": sample_id,
                                "qa_index": f"{qa_source}:{qa_name}",
                                "image": sample_image,
                            }
                        )
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if not self.tokenize:
            item = {
                "image": self._load_image(sample["image"])
                if sample["image"] is not None
                else None,
                "question": sample["question"],
                "answer": sample["answer"],
                "index": sample["index"],
                "id": sample["id"],
                "qa_index": sample["qa_index"],
            }
        else:
            item = self._process_sample(
                question=sample["question"],
                answer=sample["answer"],
                index=sample["index"],
            )
            item["id"] = sample["id"]
            item["qa_index"] = sample["qa_index"]
        if self.idk_responses is not None:
            idk_answer = random.choice(self.idk_responses)
            if not self.tokenize:
                alternate = dict(item)
                alternate["answer"] = idk_answer
            else:
                alternate = self._process_sample(
                    question=sample["question"],
                    answer=idk_answer,
                    index=sample["index"],
                )
            return {"original": item, "alternate": alternate}
        return item


class QAwithAlternateDataset(QADataset):
    def __init__(self, alternate_key, return_original=True, *args, **kwargs):
        self.alternate_key = alternate_key
        self.return_original = return_original
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        if isinstance(item, dict):
            return_item = {"original": item}
            alt_item = self._process_sample(
                question=question, answer=self.data[idx][self.alternate_key]
            )
            return_item["alternate"] = alt_item
            # return_item = [item, idk_item]
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                alt_item = self._process_sample(
                    question=question, answer=self.data[idx][self.alternate_key]
                )
                return_item["alternate"] = alt_item
                # return_item.append([sample_item, idk_item])
        return return_item if self.return_original else return_item["alternate"]
