import os
import json
import logging
import argparse
import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)

# Import datasets với try-except để xử lý nếu thiếu
try:
    from datasets import Dataset, DatasetDict
except ImportError:
    raise ImportError("Thư viện 'datasets' chưa được cài đặt. Hãy cài đặt bằng: pip install datasets")

# Import fuzzywuzzy với try-except để xử lý nếu thiếu
try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None
    logging.warning("Thư viện fuzzywuzzy chưa được cài đặt. Chức năng chuẩn hóa kỹ năng sẽ bị vô hiệu hóa. Hãy cài đặt bằng: pip install fuzzywuzzy python-Levenshtein")

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers.trainer_utils import EvalPrediction

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Định nghĩa các nhãn
LABELS = ["O", "B-SKILL", "I-SKILL"]
LABEL_MAP = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}


class NERDataProcessor:
    """Lớp xử lý dữ liệu cho mô hình NER"""

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def extract_skills_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Trích xuất kỹ năng từ phần SKILLS trong văn bản CV.
        - Tìm đoạn văn bản sau tiêu đề 'SKILLS' hoặc 'Skills'.
        - Trả về danh sách kỹ năng với vị trí (start, end) và nhãn 'SKILL'.
        """
        skills = []
        skills_section = re.search(r'(?i)SKILLS\n(.*?)(?=\n[A-Z\s]+:|\n[A-Z\s]+\n|$)', text, re.DOTALL)
        if not skills_section:
            return skills

        skills_text = skills_section.group(1).strip()
        raw_skills = [s.strip() for s in skills_text.split(',')]
        raw_skills = [s for s in raw_skills if s]

        for skill in raw_skills:
            for match in re.finditer(re.escape(skill), text):
                start, end = match.start(), match.end()
                skills.append({
                    "start": start,
                    "end": end,
                    "label": "SKILL"
                })

        return skills

    def create_dataset_from_dir(self, dir_path: str, output_jsonl: str, text_field: str = "text") -> int:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.error(f"Thư mục không tồn tại: {dir_path}")
            raise ValueError(f"Thư mục không tồn tại: {dir_path}")

        json_files = list(dir_path.glob('*.json'))
        num_files = len(json_files)
        logger.info(f"Tìm thấy {num_files} file JSON trong {dir_path}")

        if num_files == 0:
            raise ValueError("Không tìm thấy file JSON nào trong thư mục.")

        samples = []
        skipped = 0
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                text = data.get(text_field, "")
                if not text.strip():
                    logger.warning(f"Bỏ qua file {file_path.name}: Không có văn bản")
                    skipped += 1
                    continue

                entities = self.extract_skills_from_text(text)
                samples.append({
                    "text": text,
                    "entities": entities
                })

            except json.JSONDecodeError as e:
                logger.warning(f"Bỏ qua file {file_path.name}: {e}")
                skipped += 1
            except Exception as e:
                logger.warning(f"Lỗi khi đọc file {file_path.name}: {e}")
                skipped += 1

        os.makedirs(Path(output_jsonl).parent, exist_ok=True)
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        logger.info(f"Đã tạo {len(samples)} mẫu vào {output_jsonl} (bỏ qua {skipped} file)")
        return len(samples)

    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data.append(json.loads(line.strip()))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Bỏ qua dòng không hợp lệ trong {file_path}: {e}")
            logger.info(f"Đã đọc {len(data)} mẫu từ {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"File không tồn tại: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Lỗi khi đọc file {file_path}: {e}")
            raise

    def _convert_to_dataset_format(self, examples: List[Dict[str, Any]]) -> Dict[str, List]:
        input_ids = []
        attention_masks = []
        labels_list = []

        for example in examples:
            text = example.get("text", "")
            entities = example.get("entities", [])

            if not text:
                logger.warning("Bỏ qua mẫu không có văn bản")
                continue

            encoding = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
                return_tensors=None
            )

            token_ids = encoding["input_ids"]
            attn_mask = encoding["attention_mask"]
            offsets = encoding["offset_mapping"]

            label_ids = []
            for (tok_start, tok_end) in offsets:
                if tok_start == tok_end:
                    label_ids.append(-100)
                    continue

                assigned = False
                for ent in entities:
                    try:
                        if isinstance(ent, dict):
                            start = int(ent.get("start", 0))
                            end = int(ent.get("end", 0))
                            label = ent.get("label", "SKILL")
                        elif isinstance(ent, (list, tuple)) and len(ent) >= 2:
                            start = int(ent[0])
                            end = int(ent[1])
                            label = ent[2] if len(ent) > 2 else "SKILL"
                        else:
                            continue
                        if tok_start < end and tok_end > start:
                            if tok_start <= start < tok_end:
                                label_ids.append(LABEL_MAP.get(f"B-{label}", LABEL_MAP["B-SKILL"]))
                            else:
                                label_ids.append(LABEL_MAP.get(f"I-{label}", LABEL_MAP["I-SKILL"]))
                            assigned = True
                            break
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Bỏ qua thực thể không hợp lệ: {ent}, lỗi: {e}")
                        continue

                if not assigned:
                    label_ids.append(LABEL_MAP["O"])

            input_ids.append(token_ids)
            attention_masks.append(attn_mask)
            labels_list.append(label_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels_list
        }

    def prepare_dataset(self, data: List[Dict[str, Any]], split_ratio: float = 0.8) -> DatasetDict:
        processed_data = []
        for item in data:
            text = item.get("text", "")
            entities = item.get("entities", [])
            norm_entities = []
            for ent in entities:
                try:
                    if isinstance(ent, dict):
                        start = int(ent.get("start", 0))
                        end = int(ent.get("end", 0))
                        label = ent.get("label", "SKILL")
                    elif isinstance(ent, (list, tuple)) and len(ent) >= 2:
                        start = int(ent[0])
                        end = int(ent[1])
                        label = ent[2] if len(ent) > 2 else "SKILL"
                    else:
                        raise ValueError(f"Invalid entity format: {ent}")
                    if start < end:
                        norm_entities.append([start, end, label])
                except (ValueError, TypeError) as e:
                    logger.warning(f"Bỏ qua thực thể không hợp lệ: {ent}, lỗi: {e}")
                    continue
            processed_data.append({"text": text, "entities": norm_entities})

        np.random.seed(42)
        np.random.shuffle(processed_data)
        split_idx = int(len(processed_data) * split_ratio)
        train_data = processed_data[:split_idx]
        val_data = processed_data[split_idx:]

        train_features = self._convert_to_dataset_format(train_data)
        val_features = self._convert_to_dataset_format(val_data)

        dataset_dict = DatasetDict({
            "train": Dataset.from_dict(train_features),
            "validation": Dataset.from_dict(val_features)
        })
        logger.info(f"Dataset: {len(train_data)} train, {len(val_data)} validation")
        return dataset_dict


class NERTrainer:
    def __init__(self, model_name: str = "bert-base-uncased", output_dir: str = "./models/ner_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data_processor = NERDataProcessor(self.tokenizer)
        os.makedirs(output_dir, exist_ok=True)

    def train(self, dataset: DatasetDict, epochs: int = 3, batch_size: int = 16, learning_rate: float = 5e-5):
        try:
            model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=len(LABELS),
                id2label=ID_TO_LABEL,
                label2id=LABEL_MAP
            )
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình {self.model_name}: {e}")
            raise

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=100,
            save_total_limit=2,
            report_to="none"
        )

        def compute_metrics(pred: EvalPrediction):
            predictions = np.argmax(pred.predictions, axis=2)
            labels = pred.label_ids

            true_labels = [[ID_TO_LABEL[l] for l in label if l != -100] for label in labels]
            true_predictions = [
                [ID_TO_LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            all_true_labels = [label for sublist in true_labels for label in sublist]
            all_predictions = [pred for sublist in true_predictions for pred in sublist]

            precision, recall, f1, _ = precision_recall_fscore_support(
                all_true_labels, all_predictions, average="weighted", zero_division=0
            )
            acc = accuracy_score(all_true_labels, all_predictions)

            return {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

        data_collator = DataCollatorForTokenClassification(self.tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )

        logger.info("Bắt đầu huấn luyện mô hình...")
        trainer.train()

        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Đã lưu mô hình vào {self.output_dir}")

        eval_result = trainer.evaluate()
        logger.info(f"Kết quả đánh giá: {eval_result}")

        return eval_result


class SkillNormalizer:
    def __init__(self, normalization_dict_path: Optional[str] = None):
        self.normalization_dict = {}
        if normalization_dict_path and os.path.exists(normalization_dict_path):
            try:
                with open(normalization_dict_path, 'r', encoding='utf-8') as f:
                    self.normalization_dict = json.load(f)
                logger.info(f"Đã tải từ điển chuẩn hóa với {len(self.normalization_dict)} mục")
            except Exception as e:
                logger.error(f"Lỗi khi tải từ điển chuẩn hóa: {e}")
                raise

    def normalize_skill(self, skill: str, threshold: int = 80) -> str:
        if fuzz is None:
            logger.warning("fuzzywuzzy chưa cài đặt, bỏ qua chuẩn hóa và trả về skill gốc.")
            return skill.strip()

        if not skill:
            return skill

        skill = skill.strip()
        if skill in self.normalization_dict:
            return self.normalization_dict[skill]

        best_match = None
        best_score = 0
        for known_skill in self.normalization_dict:
            score = fuzz.ratio(skill.lower(), known_skill.lower())
            if score > best_score and score >= threshold:
                best_score = score
                best_match = known_skill

        return self.normalization_dict.get(best_match, skill)

    def add_to_dict(self, skill: str, normalized_skill: str):
        self.normalization_dict[skill] = normalized_skill

    def save_dict(self, output_path: str):
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.normalization_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"Đã lưu từ điển chuẩn hóa vào {output_path}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu từ điển chuẩn hóa: {e}")
            raise


class NERInference:
    def __init__(self, model_dir: str, skill_normalizer: Optional[SkillNormalizer] = None):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
            self.skill_normalizer = skill_normalizer
            self.model.eval()
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình hoặc tokenizer từ {model_dir}: {e}")
            raise

    def extract_skills(self, text: str) -> List[Dict[str, Any]]:
        try:
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
                return_offsets_mapping=True
            )

            model_inputs = {k: v.to(self.model.device) for k, v in enc.items() if k != "offset_mapping"}
            offsets = enc["offset_mapping"][0].tolist()

            with torch.no_grad():
                outputs = self.model(**model_inputs)
                pred_ids = torch.argmax(outputs.logits, dim=2)[0].tolist()

            skills = []
            current_span = None

            for tok_idx, pred_id in enumerate(pred_ids):
                if tok_idx >= len(offsets):
                    break
                start, end = offsets[tok_idx]
                if start == end:
                    continue

                label = ID_TO_LABEL.get(pred_id, "O")
                if label == "B-SKILL":
                    if current_span is not None:
                        s, e = current_span
                        skill_text = text[s:e].strip()
                        if skill_text:
                            skill_info = {"skill": skill_text, "start": s, "end": e}
                            if self.skill_normalizer:
                                skill_info["normalized"] = self.skill_normalizer.normalize_skill(skill_text)
                            skills.append(skill_info)
                    current_span = (start, end)
                elif label == "I-SKILL" and current_span is not None:
                    current_span = (current_span[0], end)
                else:
                    if current_span is not None:
                        s, e = current_span
                        skill_text = text[s:e].strip()
                        if skill_text:
                            skill_info = {"skill": skill_text, "start": s, "end": e}
                            if self.skill_normalizer:
                                skill_info["normalized"] = self.skill_normalizer.normalize_skill(skill_text)
                            skills.append(skill_info)
                        current_span = None

            if current_span is not None:
                s, e = current_span
                skill_text = text[s:e].strip()
                if skill_text:
                    skill_info = {"skill": skill_text, "start": s, "end": e}
                    if self.skill_normalizer:
                        skill_info["normalized"] = self.skill_normalizer.normalize_skill(skill_text)
                    skills.append(skill_info)

            return skills
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất kỹ năng: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện và suy luận mô hình NER cho kỹ năng")
    parser.add_argument("--model_dir", type=str, default="./models/ner_model", help="Thư mục chứa mô hình")
    parser.add_argument("--output_dir", type=str, default="./models/ner_model", help="Thư mục lưu mô hình")
    parser.add_argument("--create_dataset", action="store_true", help="Tạo dataset JSONL từ thư mục file JSON CV")
    parser.add_argument("--cv_dir", type=str, default="D:/Desktop/PycharmProjects/AI Approach/CVPDF_Parser/Clean_Text", help="Thư mục chứa file JSON CV")
    parser.add_argument("--output_jsonl", type=str, default="./data/cv_ner_dataset.jsonl", help="File JSONL đầu ra")
    parser.add_argument("--text_field", type=str, default="text", help="Tên trường chứa văn bản CV trong JSON")
    parser.add_argument("--do_train", action="store_true", help="Huấn luyện mô hình")
    parser.add_argument("--dataset", type=str, default="./data/cv_ner_dataset.jsonl", help="Đường dẫn đến tập dữ liệu")
    parser.add_argument("--epochs", type=int, default=3, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=16, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=5e-5, help="Tốc độ học")
    parser.add_argument("--normalization_dict", type=str, default=None, help="Đường dẫn đến từ điển chuẩn hóa kỹ năng")
    parser.add_argument("--do_infer", action="store_true", help="Thực hiện suy luận")
    parser.add_argument("--infer_text", type=str, default=None, help="Văn bản để trích xuất kỹ năng")

    args = parser.parse_args()

    if args.create_dataset:
        logger.info("Bắt đầu tạo dataset từ thư mục CV...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        processor = NERDataProcessor(tokenizer)
        num_samples = processor.create_dataset_from_dir(args.cv_dir, args.output_jsonl, args.text_field)
        print(f"Hoàn thành! Đã tạo {num_samples} mẫu trong {args.output_jsonl}")
        return

    if args.do_train:
        logger.info("Bắt đầu quá trình huấn luyện...")
        trainer = NERTrainer(model_name="bert-base-uncased", output_dir=args.output_dir)
        data = trainer.data_processor.load_data(args.dataset)
        dataset = trainer.data_processor.prepare_dataset(data)
        trainer.train(dataset, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)

    if args.do_infer:
        if not args.infer_text:
            logger.error("Vui lòng cung cấp văn bản để trích xuất kỹ năng (--infer_text)")
            return

        logger.info("Bắt đầu quá trình suy luận...")
        skill_normalizer = SkillNormalizer(args.normalization_dict) if args.normalization_dict else None
        inference = NERInference(args.model_dir, skill_normalizer)
        skills = inference.extract_skills(args.infer_text)

        print("\nKỹ năng được trích xuất:")
        for skill in skills:
            if "normalized" in skill:
                print(f"- {skill['skill']} (chuẩn hóa: {skill['normalized']})")
            else:
                print(f"- {skill['skill']}")


if __name__ == "__main__":
    main()
