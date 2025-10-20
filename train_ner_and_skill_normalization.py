import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# --- Helpers: chuẩn hóa đầu vào JSONL thành format huggingface datasets ---

def load_jsonl(path: str):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def convert_to_ner_examples(records: List[dict]):
    """Chuyển thành list of dicts: {"id","tokens?", "text","entities": [(start,end,label), ...]}"""
    out = []
    for i, r in enumerate(records):
        text = r.get('text') or r.get('content') or r.get('sentence')
        if text is None:
            # try other keys
            keys = list(r.keys())
            if keys:
                text = r[keys[0]]
        entities = []
        if 'entities' in r and isinstance(r['entities'], list):
            # entities might already be list of [start,end,label] or dicts
            for e in r['entities']:
                if isinstance(e, list) and len(e) >= 3:
                    start, end, label = e[0], e[1], e[2]
                elif isinstance(e, dict):
                    start = e.get('start')
                    end = e.get('end')
                    label = e.get('label') or e.get('type')
                else:
                    continue
                entities.append((start, end, label))
        elif 'labels' in r and isinstance(r['labels'], list):
            for e in r['labels']:
                start = e.get('start')
                end = e.get('end')
                label = e.get('label') or e.get('type')
                entities.append((start, end, label))
        else:
            # no entities
            entities = []
        out.append({"id": str(i), "text": text, "entities": entities})
    return out


# --- Token classification dataset conversion ---

def align_labels_with_tokens(tokenizer, text, entities, label_to_id):
    """Tokenize the text and create labels per token using BIO scheme.
    entities: list of (start, end, label) where start/end are char indexes
    Returns input_ids, attention_mask, labels (list aligned to tokens)
    """
    encoding = tokenizer(text, return_offsets_mapping=True, truncation=True)
    offsets = encoding.pop('offset_mapping')
    labels = [label_to_id['O']] * len(offsets)

    for (start, end, label) in entities:
        if start is None or end is None:
            continue
        # find token indices that overlap with this span
        token_indices = []
        for i, (s, e) in enumerate(offsets):
            if e == 0 and s == 0:
                continue  # special tokens
            if not (e <= start or s >= end):
                token_indices.append(i)
        if not token_indices:
            continue
        # assign B- and I-
        b_label = f'B-{label}'
        i_label = f'I-{label}'
        labels[token_indices[0]] = label_to_id.get(b_label, label_to_id['O'])
        for idx in token_indices[1:]:
            labels[idx] = label_to_id.get(i_label, label_to_id['O'])

    return encoding['input_ids'], encoding['attention_mask'], labels


# --- Normalization utilities ---

def load_master_skill_list(path: str) -> List[str]:
    skills = []
    if not os.path.exists(path):
        print(f"Warning: master skill list not found at {path}. Normalization will be disabled.")
        return skills
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                skills.append(s)
    return skills


def normalize_skill(raw_skill: str, master_skills: List[str], top_k=1):
    """Return best match from master_skills using RapidFuzz (fuzzy matching)"""
    try:
        from rapidfuzz import process
    except Exception:
        raise ImportError("rapidfuzz is required for normalization. Install with: pip install rapidfuzz")
    if not master_skills:
        return raw_skill, []
    matches = process.extract(raw_skill, master_skills, limit=top_k)
    # matches: list of (skill, score, index)
    return matches[0][0], matches


# --- Main: training with transformers ---

def make_label_map(entity_labels: List[str]):
    labels = ['O']
    for lab in sorted(set(entity_labels)):
        labels.append(f'B-{lab}')
        labels.append(f'I-{lab}')
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    id_to_label = {i: lab for lab, i in label_to_id.items()}
    return labels, label_to_id, id_to_label


def prepare_hf_dataset(examples, tokenizer, label_to_id):
    # Convert to huggingface-friendly dicts
    input_ids_list = []
    attention_list = []
    label_list = []
    for ex in examples:
        input_ids, attention_mask, labels = align_labels_with_tokens(tokenizer, ex['text'], ex['entities'], label_to_id)
        input_ids_list.append(input_ids)
        attention_list.append(attention_mask)
        label_list.append(labels)
    # We will use dataset.Dataset.from_dict in the main training function, so return lists
    return input_ids_list, attention_list, label_list


# A compact Trainer-based training flow

def train_transformer_ner(args):
    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
    import numpy as np
    from seqeval.metrics import classification_report

    # 1) load raw
    records = load_jsonl(args.dataset)
    examples = convert_to_ner_examples(records)

    # collect entity label set
    entity_labels = []
    for ex in examples:
        for (_, _, lab) in ex['entities']:
            if lab not in entity_labels:
                entity_labels.append(lab)

    labels, label_to_id, id_to_label = make_label_map(entity_labels)
    print('Labels:', labels)

    # 2) tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(labels))

    # 3) prepare data
    input_ids_list, attention_list, label_list = prepare_hf_dataset(examples, tokenizer, label_to_id)

    # build dataset
    ds = Dataset.from_dict({
        'input_ids': input_ids_list,
        'attention_mask': attention_list,
        'labels': label_list,
        'text': [ex['text'] for ex in examples]
    })

    # simple split
    ds = ds.train_test_split(test_size=0.1, seed=42)

    # data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # compute metrics
    def compute_metrics(p):
        predictions, labels_batch = p
        predictions = np.argmax(predictions, axis=2)
        true_labels = []
        pred_labels = []
        for pred, lab, mask in zip(predictions, labels_batch, (labels_batch != -100)):
            # convert ids to label names; note: dataset labels may have padding; here we assume labels list aligns
            tl = []
            pl = []
            for p_id, l_id in zip(pred, lab):
                if l_id == -100:
                    continue
                tl.append(id_to_label.get(int(l_id), 'O'))
                pl.append(id_to_label.get(int(p_id), 'O'))
            true_labels.append(tl)
            pred_labels.append(pl)
        # use seqeval
        report = classification_report(true_labels, pred_labels, output_dict=True)
        return {
            'precision': report.get('micro avg', {}).get('precision', 0),
            'recall': report.get('micro avg', {}).get('recall', 0),
            'f1': report.get('micro avg', {}).get('f1-score', 0)
        }

    # training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy='epoch',
        logging_strategy='steps',
        logging_steps=50,
        save_strategy='epoch',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        learning_rate=args.lr,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model='f1'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print('Training done. Model saved to', args.output_dir)


# --- Simple inference + normalization example ---

def inference_and_normalize(model_dir: str, texts: List[str], master_skill_path: str = None):
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import torch
    from rapidfuzz import process

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    labels = model.config.id2label if hasattr(model.config, 'id2label') else None
    if isinstance(labels, dict):
        id2label = {int(k): v for k, v in labels.items()}
    else:
        # try default mapping
        id2label = {i: lab for i, lab in enumerate(model.config.label2id.keys())}

    master_skills = load_master_skill_list(master_skill_path) if master_skill_path else []

    results = []
    for text in texts:
        enc = tokenizer(text, return_offsets_mapping=True, return_tensors='pt', truncation=True)
        with torch.no_grad():
            out = model(**{k: v for k, v in enc.items() if k!='offset_mapping'})
        logits = out.logits.squeeze(0).cpu().numpy()
        preds = logits.argmax(axis=-1)
        offsets = enc['offset_mapping'].squeeze(0).cpu().numpy()

        # collect spans with B-/I- scheme
        spans = []
        cur_span = None
        for pred_id, (s, e) in zip(preds, offsets):
            label = id2label.get(int(pred_id), 'O')
            if label == 'O' or e==0 and s==0:
                if cur_span:
                    spans.append(cur_span)
                    cur_span = None
                continue
            if label.startswith('B-'):
                if cur_span:
                    spans.append(cur_span)
                cur_span = {'label': label[2:], 'start': s, 'end': e}
            elif label.startswith('I-') and cur_span:
                cur_span['end'] = e
            else:
                # fallback
                if cur_span:
                    spans.append(cur_span)
                cur_span = None
        if cur_span:
            spans.append(cur_span)

        extracted = []
        for sp in spans:
            raw = text[sp['start']:sp['end']]
            canonical = raw
            best = None
            if master_skills:
                match = process.extractOne(raw, master_skills)
                if match:
                    canonical = match[0]
                    best = match
            extracted.append({'text': raw, 'label': sp['label'], 'canonical': canonical, 'match': best})
        results.append({'text': text, 'extracted': extracted})
    return results


# --- Command line interface ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/mnt/data/cv_ner_dataset.jsonl')
    parser.add_argument('--model_name', type=str, default='bert-base-multilingual-cased')
    parser.add_argument('--output_dir', type=str, default='./ner_model')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--master_skill_list', type=str, default='./skills_master.txt')
    parser.add_argument('--do_infer', action='store_true')
    parser.add_argument('--infer_text', type=str, default=None, help='If provided, run inference on this single text (in quotes)')
    args = parser.parse_args()

    if args.do_train:
        train_transformer_ner(args)

    if args.do_infer:
        texts = [args.infer_text] if args.infer_text else ["Example: Thành thạo Python, TensorFlow và quản trị Linux."]
        res = inference_and_normalize(args.output_dir, texts, master_skill_path=args.master_skill_list)
        print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
