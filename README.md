# zh-CN-Multi-Mask-Bert (CNMBert)
![image](https://github.com/user-attachments/assets/a888fde7-6766-43f1-a753-810399418bda)

---

一个用来翻译拼音缩写的模型

此模型基于[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)训练而来，通过修改其预训练任务来使其适配拼音缩写翻译任务，相较于微调过的GPT模型以及GPT-4o达到了sota

---

## 什么是拼音缩写

形如:

> "bhys" -> "不好意思"
>
> "ys" -> "原神"

这样的，使用拼音首字母来代替汉字的缩写，我们姑且称之为拼音缩写。

如果对拼音缩写感兴趣可以看看这个↓

[大家为什么会讨厌缩写？ - 远方青木的回答 - 知乎](https://www.zhihu.com/question/269016377/answer/2654824753)

### CNMBert

| Model           | 模型权重                                                    | Memory Usage (FP16) | QPS   | MRR   | Acc   |
| --------------- | ----------------------------------------------------------- | ------------------- | ----- | ----- | ----- |
| CNMBert-Default | [Huggingface](https://huggingface.co/Midsummra/CNMBert)     | 0.4GB               | 12.56 | 59.70 | 49.74 |
| CNMBert-MoE     | [Huggingface](https://huggingface.co/Midsummra/CNMBert-MoE) | 0.8GB               | 3.20  | 61.53 | 51.86 |

* 所有模型均在相同的150万条wiki以及知乎语料下训练
* QPS 为 queries per second (由于没有使用c重写predict所以现在性能很糟...)
* MRR 为平均倒数排名(mean reciprocal rank)
* Acc 为准确率(accuracy)

### Usage

```python
from transformers import AutoTokenizer, BertConfig

from CustomBertModel import fixed_predict
from MoELayer import BertWwmMoE
```

加载模型

```python
# use CNMBert with MoE
# To use CNMBert without MoE, replace all "Midsummra/CNMBert-MoE" with "Midsummra/CNMBert" and use BertForMaskedLM instead of using BertWwmMoE
tokenizer = AutoTokenizer.from_pretrained("Midsummra/CNMBert-MoE")
config = BertConfig.from_pretrained('Midsummra/CNMBert-MoE')
model = BertWwmMoE.from_pretrained('Midsummra/CNMBert-MoE', config=config).to('cuda')

# model = BertForMaskedLM.from_pretrained('Midsummra/CNMBert').to('cuda')
```

预测词语

```python
print(fixed_predict("我有两千kq", "kq", model, tokenizer)[:5])
print(fixed_predict("快去给魔理沙看b吧", "b", model, tokenizer[:5]))
```

> ['块钱', 1.2056937473156175], ['块前', 0.05837443749364857], ['开千', 0.0483869208528063], ['可千', 0.03996622172280445], ['口气', 0.037183335575008414]

> ['病', 1.6893256306648254], ['吧', 0.1642467901110649], ['呗', 0.026976384222507477], ['包', 0.021441461518406868], ['报', 0.01396679226309061]

### 如何微调模型

请参考[TrainExample.ipynb](https://github.com/IgarashiAkatuki/CNMBert/blob/main/TrainExample.ipynb),在数据集的格式上，只要保证csv的第一列为要训练的语料即可。

### Q&A

Q: 这玩意的速度太慢啦！！！

A: 已经有计划拿C重写predict了，，，



Q: 这玩意的准确度好差啊

A: 因为是在很小的数据集(200w)上进行的预训练，所以泛化能力很差很正常，，，可以在更大数据集或者更加细分的领域进行微调，具体微调方式和[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)差别不大，只需要将`DataCollactor`替换为`CustomBertModel.py`中的`DataCollatorForMultiMask`。

### 引用
如果您对CNMBert的具体实现感兴趣的话，可以参考
```
@misc{feng2024cnmbertmodelhanyupinyin,
      title={CNMBert: A Model For Hanyu Pinyin Abbreviation to Character Conversion Task}, 
      author={Zishuo Feng and Feng Cao},
      year={2024},
      eprint={2411.11770},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.11770}, 
}
```


