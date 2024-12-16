import warnings
import random
from collections import defaultdict
from typing import Optional, Union, Tuple, List, Mapping, Any, Dict
import copy
import torch
from sqlalchemy.dialects.postgresql import array
from sympy.stats.sampling.sample_numpy import numpy
from transformers.data.data_collator import _torch_collate_batch, tolist, _numpy_collate_batch, \
    DataCollatorForLanguageModeling
from transformers.modeling_outputs import MaskedLMOutput
import torch.nn.functional as F
import re
import copy



def get_custom_masks() -> dict:
    return {
        '[NUM]': 22128,
        '[LETTER_A]': 22129,
        '[LETTER_B]': 22130,
        '[LETTER_C]': 22131,
        '[LETTER_D]': 22132,
        '[LETTER_E]': 22133,
        '[LETTER_F]': 22134,
        '[LETTER_G]': 22135,
        '[LETTER_H]': 22136,
        '[LETTER_I]': 22137,
        '[LETTER_J]': 22138,
        '[LETTER_K]': 22139,
        '[LETTER_L]': 22140,
        '[LETTER_M]': 22141,
        '[LETTER_N]': 22142,
        '[LETTER_O]': 22143,
        '[LETTER_P]': 22144,
        '[LETTER_Q]': 22145,
        '[LETTER_R]': 22146,
        '[LETTER_S]': 22147,
        '[LETTER_T]': 22148,
        '[LETTER_U]': 22149,
        '[LETTER_V]': 22150,
        '[LETTER_W]': 22151,
        '[LETTER_X]': 22152,
        '[LETTER_Y]': 22153,
        '[LETTER_Z]': 22154
    }


from transformers import DataCollatorForWholeWordMask, BertTokenizer, BertTokenizerFast
from pypinyin import pinyin, lazy_pinyin, Style


class DataCollatorForWholeWordMaskWithMultipleMASKS(DataCollatorForWholeWordMask):
    # {'input_ids':
    # tensor([[ 101, 4269, 1071, 2141, 2769, 3193, 2218, 3221, 3189, 3315,  782,  749,
    #           102,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0]]),
    # 'chinese_ref': [[3, 6, 9]],
    # 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
    def __call__(self, examples):
        batch = super().__call__(examples)
        # 自动生成 attention_mask
        batch['attention_mask'] = (batch['input_ids'] != self.tokenizer.pad_token_id).long()
        for i in range(len(batch['input_ids'])):
            text = self.tokenizer.decode(batch['input_ids'][i])
            labels = batch['labels'][i]
            masked_tokens = torch.where(batch['input_ids'][i] == 103)[0]
            words = []
            index = 0
            for j in range(len(masked_tokens)):
                if j == 0:
                    words.append(self.tokenizer.decode(labels[int(masked_tokens[j])]))
                    continue
                if masked_tokens[j - 1] + 1 == masked_tokens[j]:
                    words[index] += self.tokenizer.decode(labels[int(masked_tokens[j])])
                else:
                    index += 1
                    words.append(self.tokenizer.decode(labels[int(masked_tokens[j])]))
            index = 0
            for word in words:
                first_letter = pinyin(word, style=Style.FIRST_LETTER)
                for letter in first_letter:
                    if index >= len(masked_tokens):
                        break
                    if len(letter) == 0 or letter[0] == ' ':
                        print(1)
                        continue
                    if letter == '## ':
                        # index += 1
                        continue
                    if str.islower(letter[0][0]):
                        batch['input_ids'][i][masked_tokens[index]] = 1 - 97 + ord(letter[0][0])
                    else:
                        batch['input_ids'][i][masked_tokens[index]] = 103
                    index += 1
                    if index >= len(masked_tokens):
                        break
            # print(len(batch['input_ids'][i]))
        return batch


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
            if 2 <= len(val) < 3:
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
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
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
            print(replaced_index)
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

        # 10% of the time, we replace masked input tokens with random word
        # indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        # inputs[indices_random] = random_words[indices_random]

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


from torch.nn import CrossEntropyLoss
import numpy as np

from transformers import BertForMaskedLM
import pkuseg
seg = pkuseg.pkuseg()

class MultiMaskedLM(BertForMaskedLM):
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        # print(prediction_scores)
        masked_lm_loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def predict(sentence: str,
            predict_word: str,
            model,
            tokenizer,
            top_k=10):
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
    if index == -1 or len(predict_word) >= 3 or str.isdigit(predict_word):
        # print(1)
        sentence = sentence.replace(predict_word, '[MASK][MASK]')
        inputs = tokenizer(sentence, max_length=512,
                           padding='max_length',
                           truncation=True,
                           return_tensors='pt').to('cuda')
    else:
        try:
            inputs['input_ids'][0][index + 1:index + 1 + len(replaced_word)] = torch.tensor(replaced_word).to('cuda')
        except Exception:
            print('error')
            return []
    # sentence.find(predict_word)
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


    # if len(top_k_tokens) == 0:
    #     print('不存在mask')
    #     return
    # # 解码 top-k tokens 并输出它们的概率
    # for i in range(top_k):
    #     tokens = []
    #     probs = 0.0
    #     for j in range(len(top_k_tokens[:, i])):
    #         tokens.append(top_k_tokens[j, i])
    #         probs += top_k_probs[j, i].item()
    #     predicted_token = tokenizer.decode(tokens)
    #     probs /= len(top_k_tokens[:, i])
    #     print(f"Predicted token: {predicted_token}, Probability: {probs:.4f}")
    results = []
    dfs(results=results,
        depth=0,
        probability=1.0,
        top_k_probs=top_k_probs,
        top_k_tokens=top_k_tokens,
        tokenizer=tokenizer,
        sentence=[],
        top_k=top_k)
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def dfs(results: list,
        depth: int,
        probability: float,
        top_k_tokens,
        top_k_probs,
        tokenizer,
        sentence: list,
        top_k: int=10):

    # total_predict = []
    # for i in range(len(top_k_probs[:, 0])):
    #     total_predict.append(torch.sum(top_k_probs[i, :]).tolist())

    for i in range(top_k):

        token = top_k_tokens[depth, i]
        decoded_token = tokenizer.decode(token)
        sentence.append(decoded_token)  # 使用列表来管理句子

        current_prob = top_k_probs[depth, i].item()
        new_probability = probability * current_prob  # 累加每个token的概率

        # 如果达到了最后一层深度，记录结果
        if depth == len(top_k_tokens[:, 0]) - 1:
            if new_probability >= 0.00:
                if len(seg.cut(''.join(sentence))) <= 1:
                    results.append([''.join(sentence), new_probability])
                # else:
                # cnmbert-default.append([''.join(sentence), new_probability])
                # if len(cnmbert-default) >= top_k:
                #     return
        else:
            # 继续递归到下一层
            dfs(results,
                depth=depth+1,
                probability=new_probability,
                top_k_tokens=top_k_tokens,
                top_k_probs=top_k_probs,
                tokenizer=tokenizer,
                sentence=sentence,
                top_k=top_k)

        # 回溯时撤销添加的token
        sentence.pop()

def fixed_predict(sentence: str,
            predict_word: str,
            model,
            tokenizer,
            top_k=10):
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

    # with torch.no_grad():
    #     logits = model(**inputs).logits
    #
    # # retrieve index of [MASK]
    # mask_token_index = []
    # temp = torch.nonzero(inputs['input_ids'] == 103, as_tuple=True)
    # if len(temp[0]) > 0:
    #     mask_token_index.extend(temp[1].tolist())
    # for i in range(28):
    #     temp = torch.nonzero(inputs['input_ids'] == (i + 1), as_tuple=True)
    #     if len(temp[0]) > 0:
    #         mask_token_index.extend(temp[1].tolist())
    # mask_token_index = sorted(mask_token_index)
    #
    # mask_token_logits = logits[0, mask_token_index, :]
    # mask_token_probs = F.softmax(mask_token_logits, dim=-1)
    # top_k_probs, top_k_tokens = torch.topk(mask_token_probs, top_k, dim=1)

    results_a = []
    results_b = []
    fixed_dfs(results=results_a,
        depth=length-1,
        probability=1.0,
        tokenizer=tokenizer,
        sentence=[],
        inputs=copy.deepcopy(inputs),
        model=model,
        index=index,
        top_k=top_k)
    fixed_dfs_back(results=results_b,
        depth=length-1,
        probability=1.0,
        tokenizer=tokenizer,
        sentence=[],
        inputs=copy.deepcopy(inputs),
        model=model,
        index=index + length,
        top_k=top_k)
    result_dict = defaultdict(int)
    for val in results_a + results_b:
        key = val[0]
        value = val[1]
        result_dict[key] += value
    results = [[key, val] for key, val in result_dict.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def fixed_dfs(results: list,
              depth: int,
              probability: float,
              tokenizer,
              sentence: list,
              inputs: str,
              model,
              index: int,
              top_k: int=5
              ):

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
    token = tokenizer.decode(top_k_tokens[0, 0:top_k]).split(' ')
    for i in range(len(token)):
        sentence.append(token[i])
        prob = top_k_probs[0, i].item()
        new_probability = probability * prob

        if depth == 0:
            if new_probability >= 0.00:
                if len(seg.cut(''.join(sentence))) <= 1:
                    results.append([''.join(sentence), new_probability])
        else:
            inputs['input_ids'][0][index+1] = top_k_tokens[0, i]
            fixed_dfs(results=results,
                      depth=depth-1,
                      probability=new_probability,
                      tokenizer=tokenizer,
                      sentence=sentence,
                      inputs=copy.deepcopy(inputs),
                      model=model,
                      index=index+1,
                      top_k=top_k-2)
        sentence.pop()



def fixed_dfs_back(results: list,
              depth: int,
              probability: float,
              tokenizer,
              sentence: list,
              inputs: str,
              model,
              index: int,
              top_k: int=5
              ):

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
                if len(seg.cut(''.join(sentence))) <= 1:
                    results.append([''.join(sentence), new_probability])
        else:
            inputs['input_ids'][0][index] = top_k_tokens[-1, i]
            fixed_dfs_back(results=results,
                      depth=depth-1,
                      probability=new_probability,
                      tokenizer=tokenizer,
                      sentence=sentence,
                      inputs=copy.deepcopy(inputs),
                      model=model,
                      index=index-1,
                      top_k=top_k-2)
        sentence.pop(0)


def get_expert_index(sentence: str,
           predict_word: str,
            model,
            tokenizer,
            top_k=10):
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
        inputs['input_ids'][0][index + 1:index + 1 + len(replaced_word)] = torch.tensor(replaced_word).to('cuda')
    except Exception:
        print('error')
        return []

    with torch.no_grad():
        logits = model(**inputs).logits
