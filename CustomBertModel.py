import math
import warnings
import random
from copy import deepcopy

import torch
import torch.nn.functional as F
import re
import copy
import numpy as np
import pkuseg

from collections import defaultdict
from typing import Union, Tuple, List, Mapping, Any, Dict

from numpy.random import shuffle
from transformers.data.data_collator import _torch_collate_batch, tolist, _numpy_collate_batch, \
    DataCollatorForLanguageModeling
from transformers import BertTokenizer, BertTokenizerFast
from pypinyin import pinyin, Style


class DataCollatorForMultiMask(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling that masks entire words.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling

    <Tip>

    This collator relies on details of the implementation of subword tokenization by [`BertTokenizer`], specifically
    that subword tokens are prefixed with *##*. For tokenizers that do not adhere to this scheme, this collator will
    produce an output that is roughly equivalent to [`.DataCollatorForLanguageModeling`].

    </Tip>"""

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], Mapping):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                if id == 0:
                    continue
                ref_tokens.append(token)

            # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            mask_labels.append(self._whole_word_mask(ref_tokens))
        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        return {"input_ids": inputs, "labels": labels, 'attention_mask': attention_mask}

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], Mapping):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _numpy_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            mask_labels.append(self._whole_word_mask(ref_tokens))
        batch_mask = _numpy_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.numpy_mask_tokens(batch_input, batch_mask)
        return {"input_ids": inputs, "labels": labels}

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information."
            )

        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        weighted_cand_indexes = cand_indexes.copy()
        # 我们想让模型在进行mask操作时，更多的选择词组，而不是单个汉字，因此增加词组的权重
        for val in cand_indexes:
            # 如果是一个词组，则增加一个该词组的拷贝进入列表以增加权重
            if 2 <= len(val) <= 3:
                weighted_cand_indexes.append(val.copy())

        random.shuffle(weighted_cand_indexes)
        num_to_predict = min(max_predictions, max(2, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in weighted_cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            if index_set[0] - 1 in masked_lms or index_set[-1] + 1 in masked_lms:
                flag = random.random() > 0.5
                if flag:
                    continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def torch_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        # inputs [batch_size, max_len]
        # mask_labels [batch_size, max_len]
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        import torch

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        try:
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        except Exception:
            print(special_tokens_mask)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 90% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices

        # print(replaced_index)
        for index in range(len(inputs)):
            replaced_index = [i for i, v in enumerate(indices_replaced[index]) if v]
            # print(replaced_index)
            # 修改mask逻辑，使用多种mask替代原有的单一[MASK]标签，并剔除无效[MASK]如标点符号
            indices_total = torch.nonzero(labels[index] != -100, as_tuple=True)[0].tolist()
            values = labels[index][indices_total]
            # temp = indices_total.copy()
            indices_total = [x for i, x in enumerate(indices_total) if not ((200 <= values[i] <= 209)
                                                                            or (345 <= values[i] <= 532)
                                                                            or (106 <= values[i] <= 120)
                                                                            or (131 <= values[i] <= 142)
                                                                            or values[i] == 8024)]
            values = labels[index][indices_total]
            indices_total = torch.tensor(indices_total)
            words = []
            pattern = re.compile(r'[\u4e00-\u9fff]')
            for i, val in enumerate(indices_total):
                if i == 0:
                    words.append([values[i]])
                    continue
                # print(self.tokenizer.decode(values[i]).replace('#', '').replace(' ', ''))
                temp_word = self.tokenizer.decode(values[i]).replace('#', '').replace(' ', '')
                if str.isdigit(temp_word) or temp_word == '°' or str.isascii(temp_word):
                    words.append([values[i]])
                    continue
                if indices_total[i] == indices_total[i - 1] + 1 and pattern.search(temp_word):
                    words[-1].append(values[i])
                else:
                    words.append([values[i]])

            # [LETTER_A] ~ [LETTER_Z] = 1 ~ 26
            # [NUM] = 28
            # [SPECIAL] = 29
            mask_ids = []
            for word in words:
                word = self.tokenizer.decode(word).replace(' ', '').replace('#', '')
                first_letter = pinyin(word, style=Style.FIRST_LETTER)
                for letter in first_letter:
                    if str.islower(letter[0][0]):
                        if 1 <= (ord(letter[0][0]) - 96) <= 26:
                            mask_ids.append(ord(letter[0][0]) - 96)
                        else:
                            mask_ids.append(29)
                    else:
                        mask_ids.append(28)

            try:
                replaced_mask = []
                replaced_mask_index = []
                for i, val in enumerate(indices_total):
                    if val in replaced_index:
                        replaced_mask_index.append(val.tolist())
                        # if i >= len(mask_ids):
                        #     continue
                        replaced_mask.append(mask_ids[i])

                if len(replaced_mask_index) != 0:
                    inputs[index][replaced_mask_index] = torch.tensor(replaced_mask)
            except Exception:
                print(mask_ids)
                print(indices_total)

        return inputs, labels

    def numpy_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = np.copy(inputs)
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        masked_indices = mask_labels.astype(bool)

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        masked_indices[np.array(special_tokens_mask, dtype=bool)] = 0
        if self.tokenizer._pad_token is not None:
            padding_mask = labels == self.tokenizer.pad_token_id
            masked_indices[padding_mask] = 0

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        indices_random = (
                np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced
        )
        random_words = np.random.randint(low=0, high=len(self.tokenizer), size=labels.shape, dtype=np.int64)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class DataCollatorForDetector(DataCollatorForLanguageModeling):
    """
    Data collator used for language modeling that masks entire words.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling

    <Tip>

    This collator relies on details of the implementation of subword tokenization by [`BertTokenizer`], specifically
    that subword tokens are prefixed with *##*. For tokenizers that do not adhere to this scheme, this collator will
    produce an output that is roughly equivalent to [`.DataCollatorForLanguageModeling`].

    </Tip>"""

    def __random_token(self, current_token: int):
        random_token = current_token
        while random_token == current_token or random_token == 27:
            random_token = random.randint(1, 29)
        return random_token

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], Mapping):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                if id == 0:
                    continue
                ref_tokens.append(token)

            # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            mask_labels.append(self._whole_word_mask(ref_tokens))
        batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        return {"input_ids": inputs, "labels": labels, 'attention_mask': attention_mask}

    def numpy_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(examples[0], Mapping):
            input_ids = [e["input_ids"] for e in examples]
        else:
            input_ids = examples
            examples = [{"input_ids": e} for e in examples]

        batch_input = _numpy_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)

        mask_labels = []
        for e in examples:
            ref_tokens = []
            for id in tolist(e["input_ids"]):
                token = self.tokenizer._convert_id_to_token(id)
                ref_tokens.append(token)

            # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
            if "chinese_ref" in e:
                ref_pos = tolist(e["chinese_ref"])
                len_seq = len(e["input_ids"])
                for i in range(len_seq):
                    if i in ref_pos:
                        ref_tokens[i] = "##" + ref_tokens[i]
            mask_labels.append(self._whole_word_mask(ref_tokens))
        batch_mask = _numpy_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        inputs, labels = self.numpy_mask_tokens(batch_input, batch_mask)
        return {"input_ids": inputs, "labels": labels}

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        if not isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            warnings.warn(
                "DataCollatorForWholeWordMask is only suitable for BertTokenizer-like tokenizers. "
                "Please refer to the documentation for more information."
            )

        cand_indexes = []
        for i, token in enumerate(input_tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        weighted_cand_indexes = cand_indexes.copy()
        # 我们想让模型在进行mask操作时，更多的选择词组，而不是单个汉字，因此增加词组的权重
        for val in cand_indexes:
            # 如果是一个词组，则增加一个该词组的拷贝进入列表以增加权重
            if 2 <= len(val) <= 3:
                weighted_cand_indexes.append(val.copy())
                weighted_cand_indexes.append(val.copy())

        random.shuffle(weighted_cand_indexes)
        num_to_predict = min(max_predictions, max(2, int(round(len(input_tokens) * self.mlm_probability))))
        masked_lms = []
        covered_indexes = set()
        for index_set in weighted_cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            if index_set[0] - 1 in masked_lms or index_set[-1] + 1 in masked_lms:
                flag = random.random() > 0.5
                if flag:
                    continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        if len(covered_indexes) != len(masked_lms):
            raise ValueError("Length of covered_indexes is not equal to length of masked_lms.")
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def torch_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        # inputs [batch_size, max_len]
        # mask_labels [batch_size, max_len]
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        import torch

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        try:
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        except Exception:
            print(special_tokens_mask)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices

        # print(replaced_index)
        for index in range(len(inputs)):
            replaced_index = [i for i, v in enumerate(indices_replaced[index]) if v]
            # print(replaced_index)
            # 修改mask逻辑，使用多种mask替代原有的单一[MASK]标签，并剔除无效[MASK]如标点符号
            indices_total = torch.nonzero(labels[index] != -100, as_tuple=True)[0].tolist()
            values = labels[index][indices_total]
            # temp = indices_total.copy()
            indices_total = [x for i, x in enumerate(indices_total) if not ((200 <= values[i] <= 209)
                                                                            or (345 <= values[i] <= 532)
                                                                            or (106 <= values[i] <= 120)
                                                                            or (131 <= values[i] <= 142)
                                                                            or values[i] == 8024)]
            values = labels[index][indices_total]
            indices_total = torch.tensor(indices_total)
            words = []
            pattern = re.compile(r'[\u4e00-\u9fff]')
            for i, val in enumerate(indices_total):
                if i == 0:
                    words.append([values[i]])
                    continue
                # print(self.tokenizer.decode(values[i]).replace('#', '').replace(' ', ''))
                temp_word = self.tokenizer.decode(values[i]).replace('#', '').replace(' ', '')
                if str.isdigit(temp_word) or temp_word == '°' or str.isascii(temp_word):
                    words.append([values[i]])
                    continue
                if indices_total[i] == indices_total[i - 1] + 1 and pattern.search(temp_word):
                    words[-1].append(values[i])
                else:
                    words.append([values[i]])

            # [LETTER_A] ~ [LETTER_Z] = 1 ~ 26
            # [NUM] = 28
            # [SPECIAL] = 29
            mask_ids = []
            for word in words:
                word = self.tokenizer.decode(word).replace(' ', '').replace('#', '')
                first_letter = pinyin(word, style=Style.FIRST_LETTER)
                for letter in first_letter:
                    if str.islower(letter[0][0]):
                        if 1 <= (ord(letter[0][0]) - 96) <= 26:
                            mask_ids.append(ord(letter[0][0]) - 96)
                        else:
                            mask_ids.append(29)
                    else:
                        mask_ids.append(28)

            try:
                replaced_mask = []
                replaced_mask_index = []
                for i, val in enumerate(indices_total):
                    if val in replaced_index:
                        replaced_mask_index.append(val.tolist())
                        # if i >= len(mask_ids):
                        #     continue
                        replaced_mask.append(mask_ids[i])

                labels[index] = torch.full(labels[index].shape, 2)
                if len(replaced_mask_index) != 0:
                    word_indices = []
                    for i, val in enumerate(replaced_mask_index):
                        if i == 0:
                            word_indices.append([val])
                            continue
                        if val == word_indices[-1][-1] + 1:
                            word_indices[-1].append(val)
                        else:
                            word_indices.append([val])

                    random_replaced_mask = deepcopy(replaced_mask)
                    for word in word_indices:
                        if random.random() < 0.5:
                            for i, val in enumerate(word):
                                random_replaced_mask[i] = self.__random_token(random_replaced_mask[i])
                            labels[index][word] = torch.ones_like(torch.tensor(word))
                        else:
                            labels[index][word] = torch.zeros_like(torch.tensor(word))
                    inputs[index][replaced_mask_index] = torch.tensor(random_replaced_mask)
            except Exception:
                print("error")
                print(mask_ids)
                print(indices_total)
        # print(inputs, labels)
        return inputs, labels


    def numpy_mask_tokens(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = np.copy(inputs)
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        masked_indices = mask_labels.astype(bool)

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        masked_indices[np.array(special_tokens_mask, dtype=bool)] = 0
        if self.tokenizer._pad_token is not None:
            padding_mask = labels == self.tokenizer.pad_token_id
            masked_indices[padding_mask] = 0

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        indices_random = (
                np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced
        )
        random_words = np.random.randint(low=0, high=len(self.tokenizer), size=labels.shape, dtype=np.int64)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


__seg = pkuseg.pkuseg()

def predict(sentence: str,
            predict_word: str,
            model,
            tokenizer,
            top_k=10,
            beam_size=16,
            threshold=0.000,
            fast_mode=True,
            strict_mode=True):
    replaced_word = []
    for letter in predict_word:
        if str.isdigit(letter):
            replaced_word.append(27)
            continue
        id = ord(letter) - 96
        if 1 <= id <= 26:
            replaced_word.append(id)
        else:
            replaced_word.append(28)

    inputs = tokenizer(sentence, max_length=64,
                       padding='max_length',
                       truncation=True,
                       return_tensors='pt').to('cuda')
    index = sentence.find(predict_word)
    length = len(predict_word)

    try:
        inputs['input_ids'][0][index + 1:index + 1 + length] = torch.tensor(replaced_word).to('cuda')
    except Exception:
        print('error')
        return []

    results_a = []
    results_b = []
    __beam_search(results_a,
                beam_size=beam_size,
                max_depth=length,
                tokenizer=tokenizer,
                inputs=copy.deepcopy(inputs),
                model=model,
                top_k=top_k,
                threshold=threshold,
                index=index,
                strict_mode=strict_mode)

    if not fast_mode:
        __beam_search_back(results_b,
                         beam_size=beam_size,
                         max_depth=length,
                         tokenizer=tokenizer,
                         inputs=copy.deepcopy(inputs),
                         model=model,
                         top_k=top_k,
                         threshold=threshold,
                         index=index + length,
                         strict_mode=strict_mode)
    result_dict = defaultdict(int)
    for val in results_a + results_b:
        key = val[0]
        value = val[1]
        result_dict[key] += value
    results = [[key, val] for key, val in result_dict.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def backtrack_predict(sentence: str,
            predict_word: str,
            model,
            tokenizer,
            top_k=10,
            fast_mode=True,
            strict_mode=True):
    replaced_word = []
    for letter in predict_word:
        if str.isdigit(letter):
            replaced_word.append(27)
            continue
        id = ord(letter) - 96
        if 1 <= id <= 26:
            replaced_word.append(id)
        else:
            replaced_word.append(28)

    inputs = tokenizer(sentence, max_length=64,
                       padding='max_length',
                       truncation=True,
                       return_tensors='pt').to('cuda')
    index = sentence.find(predict_word)
    length = len(predict_word)

    try:
        inputs['input_ids'][0][index + 1:index + 1 + length] = torch.tensor(replaced_word).to('cuda')
    except Exception:
        print('error')
        return []

    results_a = []
    results_b = []
    __fixed_dfs(results=results_a,
        depth=length-1,
        probability=1.0,
        tokenizer=tokenizer,
        sentence=[],
        inputs=copy.deepcopy(inputs),
        model=model,
        index=index,
        top_k=top_k,
        strict_mode=strict_mode)

    if not fast_mode:
        __fixed_dfs_back(results=results_b,
            depth=length-1,
            probability=1.0,
            tokenizer=tokenizer,
            sentence=[],
            inputs=copy.deepcopy(inputs),
            model=model,
            index=index + length,
            top_k=top_k,
            strict_mode=strict_mode)
    result_dict = defaultdict(int)
    for val in results_a + results_b:
        key = val[0]
        value = val[1]
        result_dict[key] += value
    results = [[key, val] for key, val in result_dict.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def __fixed_dfs(results: list,
              depth: int,
              probability: float,
              tokenizer,
              sentence: list,
              inputs: str,
              model,
              index: int,
              top_k: int=5,
              strict_mode=True):

    with torch.no_grad():
        logits = model(**inputs).logits

    # retrieve index of [MASK]
    mask_token_index = torch.where((inputs['input_ids'] == 103) |
                                   ((inputs['input_ids'] >= 1) &
                                    (inputs['input_ids'] <= 28)))[1].tolist()
    mask_token_logits = logits[0, mask_token_index, :]
    mask_token_probs = F.softmax(mask_token_logits, dim=-1)
    top_k_probs, top_k_tokens = torch.topk(mask_token_probs, top_k, dim=1)
    token = tokenizer.decode(top_k_tokens[0, 0:top_k]).split(' ')
    for i in range(len(token)):
        sentence.append(token[i])
        prob = top_k_probs[0, i].item()
        new_probability = probability * prob

        if depth == 0:
            if new_probability >= 0.00:
                if not strict_mode or len(__seg.cut(''.join(sentence))) <= max(len(sentence) - 1, 1):
                    results.append([''.join(sentence), new_probability])
        else:
            original_value = torch.clone(inputs['input_ids'][0][index + 1])
            inputs['input_ids'][0][index + 1] = top_k_tokens[0, i]
            __fixed_dfs(results=results,
                      depth=depth-1,
                      probability=new_probability,
                      tokenizer=tokenizer,
                      sentence=sentence,
                      inputs=inputs,
                      model=model,
                      index=index+1,
                      top_k=top_k-2)
            inputs['input_ids'][0][index + 1] = original_value
        sentence.pop()



def __fixed_dfs_back(results: list,
              depth: int,
              probability: float,
              tokenizer,
              sentence: list,
              inputs: str,
              model,
              index: int,
              top_k: int=5,
              strict_mode=True):

    with torch.no_grad():
        logits = model(**inputs).logits

    # retrieve index of [MASK]
    mask_token_index = []
    temp = torch.nonzero(inputs['input_ids'] == 103, as_tuple=True)
    if len(temp[0]) > 0:
        mask_token_index.extend(temp[1].tolist())
    for i in range(28):
        temp = torch.nonzero(inputs['input_ids'] == (i + 1), as_tuple=True)
        if len(temp[0]) > 0:
            mask_token_index.extend(temp[1].tolist())
    mask_token_index = sorted(mask_token_index)
    mask_token_logits = logits[0, mask_token_index, :]
    mask_token_probs = F.softmax(mask_token_logits, dim=-1)
    top_k_probs, top_k_tokens = torch.topk(mask_token_probs, top_k, dim=1)
    token = tokenizer.decode(top_k_tokens[-1, 0:top_k]).split(' ')

    for i in range(len(token)):
        sentence.insert(0, token[i])
        prob = top_k_probs[-1, i].item()
        new_probability = probability * prob
        if depth == 0:
            if new_probability >= 0.00:
                if not strict_mode or len(__seg.cut(''.join(sentence))) <= max(len(sentence) - 1, 1):
                    results.append([''.join(sentence), new_probability])
        else:
            inputs['input_ids'][0][index] = top_k_tokens[-1, i]
            __fixed_dfs_back(results=results,
                      depth=depth-1,
                      probability=new_probability,
                      tokenizer=tokenizer,
                      sentence=sentence,
                      inputs=copy.deepcopy(inputs),
                      model=model,
                      index=index-1,
                      top_k=top_k-2)
        sentence.pop(0)

def __beam_search_back(results: list,
                beam_size: int,
                max_depth: int,
                tokenizer,
                inputs: str,
                model,
                top_k: int=5,
                threshold: float=0.01,
                index: int=0,
                strict_mode=True):
    beams = [[inputs, [], 1.0, index]]

    for depth in range(max_depth):
        new_beams = []

        for inputs, sentence, probability, index in beams:
            with torch.no_grad():
                logits = model(**inputs).logits

            mask_token_index = torch.where(
                (inputs['input_ids'] == 103) |
                ((inputs['input_ids'] >= 1) & (inputs['input_ids'] <= 28))
            )[1].tolist()

            if not mask_token_index:
                continue

            mask_token_logits = logits[0, mask_token_index, :]
            mask_token_probs = F.softmax(mask_token_logits, dim=-1)
            top_k_probs, top_k_tokens = torch.topk(mask_token_probs, top_k, dim=1)

            tokens = tokenizer.decode(top_k_tokens[-1, 0:top_k]).split(' ')
            for i in range(len(tokens)):
                new_probability = probability * top_k_probs[-1, i].item()

                if new_probability < threshold:
                    continue

                new_inputs = copy.deepcopy(inputs)
                new_inputs['input_ids'][0][index] = top_k_tokens[-1, i]

                new_sentence = sentence.copy()
                new_sentence.insert(0, tokens[i])
                new_beams.append([new_inputs, new_sentence, new_probability, index - 1])
        if len(new_beams) > 0:
            softmax(new_beams)
        new_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_size]

        if not new_beams:
            break

        if top_k > 2:
            top_k -= 2
        beams = new_beams

    for _, sentence, probability, _ in beams:
        sentence = ''.join(sentence)
        if not strict_mode or len(__seg.cut(sentence)) <= max(len(sentence) - 1, 1):
            results.append((sentence, probability))

    return results

def __beam_search(results: list,
                beam_size: int,
                max_depth: int,
                tokenizer,
                inputs: str,
                model,
                top_k: int=5,
                threshold: float=0.01,
                index: int=0,
                strict_mode=True):
    beams = [[inputs, [], 1.0, index]]

    for depth in range(max_depth):
        new_beams = []

        for inputs, sentence, probability, index in beams:
            with torch.no_grad():
                logits = model(**inputs).logits

            mask_token_index = torch.where(
                (inputs['input_ids'] == 103) |
                ((inputs['input_ids'] >= 1) & (inputs['input_ids'] <= 28))
            )[1].tolist()

            if not mask_token_index:
                continue

            mask_token_logits = logits[0, mask_token_index, :]
            mask_token_probs = F.softmax(mask_token_logits, dim=-1)
            top_k_probs, top_k_tokens = torch.topk(mask_token_probs, top_k, dim=1)

            tokens = tokenizer.decode(top_k_tokens[0, 0:top_k]).split(' ')
            for i in range(len(tokens)):
                new_probability = probability * top_k_probs[0, i].item()

                if new_probability < threshold:
                    continue

                new_inputs = copy.deepcopy(inputs)
                new_inputs['input_ids'][0][index + 1] = top_k_tokens[0, i]

                new_sentence = sentence + [tokens[i]]
                new_beams.append([new_inputs, new_sentence, new_probability, index + 1])
        if len(new_beams) > 0:
            softmax(new_beams)
        new_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_size]

        if not new_beams:
            break

        if top_k > 2:
            top_k -= 2
        beams = new_beams

    for _, sentence, probability, _ in beams:
        sentence = ''.join(sentence)
        if not strict_mode or len(__seg.cut(sentence)) <= max(len(sentence) - 1, 1):
            results.append((sentence, probability))

    return results

def softmax(beams: List):
    column_2 = [row[2] for row in beams]

    max_val = max(column_2)
    exp_values = [math.exp(v - max_val) for v in column_2]
    sum_exp = sum(exp_values)
    softmax_column =  [v / sum_exp for v in exp_values]

    for i, row in enumerate(beams):
        row[2] = softmax_column[i]

import Levenshtein

def word_level_predict(sentence: str,
            predict_word: str,
            model,
            tokenizer,
            top_k=10,
            beam_size=16,
            threshold=0.000,
            fast_mode=True,
            strict_mode=True):
    if predict_word.isascii():
        return predict(sentence=sentence, predict_word=predict_word, model=model, tokenizer=tokenizer, top_k=top_k, threshold=threshold, fast_mode=fast_mode, beam_size=beam_size, strict_mode=strict_mode)
    abbr_pinyin = []
    full_pinyin = []
    for character in predict_word:
        if character.isascii():
            abbr_pinyin.append(character)
            full_pinyin.append(character)
        else:
            abbr_pinyin.append(pinyin(character, Style.FIRST_LETTER)[0][0])
            full_pinyin.append(pinyin(character, Style.NORMAL)[0][0])
    abbr_pinyin = ''.join(abbr_pinyin)
    full_pinyin = ''.join(full_pinyin)
    sentence = sentence.replace(predict_word, abbr_pinyin)
    result = predict(sentence=sentence, predict_word=abbr_pinyin, model=model, tokenizer=tokenizer, top_k=top_k, threshold=threshold, fast_mode=fast_mode, beam_size=beam_size, strict_mode=strict_mode)

    if fast_mode:
        return result
    else:
        for val in result:
            word_pinyin = []
            for temp in pinyin(val[0], Style.NORMAL):
                word_pinyin.append(temp[0])
            word_pinyin = ''.join(word_pinyin)
            val.append(Levenshtein.ratio(word_pinyin, full_pinyin))
        result.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return result
