# zh-CN-Multi-Mask-Bert (CNMBert🍋)
~~吃柠檬Bert~~
![image](https://github.com/user-attachments/assets/a888fde7-6766-43f1-a753-810399418bda)

---

一个用来翻译拼音缩写的模型

此模型基于[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)训练而来，通过修改其预训练任务来使其适配拼音缩写翻译任务，相较于微调过的GPT模型以及GPT-4o达到了sota

---

## 什么是拼音缩写

形如:

> "bhys" -> "不好意思"
>
> "kb" -> "看病"

这样的，使用拼音首字母来代替汉字的缩写，我们姑且称之为拼音缩写。

如果对拼音缩写感兴趣可以看看这个↓

[大家为什么会讨厌缩写？ - 远方青木的回答 - 知乎](https://www.zhihu.com/question/269016377/answer/2654824753)

### CNMBert

| Model           | 模型权重                                                    | Memory Usage (FP16) | Model Size | QPS   | MRR   | Acc   |
| --------------- | ----------------------------------------------------------- | ------------------- | ---------- | ----- | ----- | ----- |
| CNMBert-Default* | [Huggingface](https://huggingface.co/Midsummra/CNMBert)     | 0.4GB               | 131M       | 12.56 | 59.70 | 49.74 |
| CNMBert-MoE     | [Huggingface](https://huggingface.co/Midsummra/CNMBert-MoE) | 0.8GB               | 329M       | 3.20  | 61.53 | 51.86 |

* 所有模型均在相同的200万条wiki以及知乎语料下训练
* QPS 为 queries per second 
* MRR 为平均倒数排名(mean reciprocal rank)
* Acc 为准确率(accuracy)
* CNMBert-Default 存在[量化版本](https://huggingface.co/mradermacher/CNMBert-GGUF)

模型架构&性能对比:
![overall (1)](https://github.com/user-attachments/assets/cf9575c4-c37d-484b-8a3b-f8f536ca78c9)
![output](https://github.com/user-attachments/assets/3de2b56d-f8cb-40f1-8ffa-68968bbd2ed5)


### Usage

```python
from transformers import AutoTokenizer, BertConfig

from CustomBertModel import predict
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
print(predict("我有两千kq", "kq", model, tokenizer)[:5])
print(predict("快去给魔理沙看b吧", "b", model, tokenizer[:5]))
```

> ['块钱', 1.2056937473156175], ['块前', 0.05837443749364857], ['开千', 0.0483869208528063], ['可千', 0.03996622172280445], ['口气', 0.037183335575008414]

> ['病', 1.6893256306648254], ['吧', 0.1642467901110649], ['呗', 0.026976384222507477], ['包', 0.021441461518406868], ['报', 0.01396679226309061]

---

```python
# 默认的predict函数使用束搜索
def predict(sentence: str, 
            predict_word: str,
            model,
            tokenizer,
            top_k=8,
            beam_size=16, # 束宽
            threshold=0.005, # 阈值
            fast_mode=True, # 是否使用快速模式
            strict_mode=True): # 是否对输出结果进行检查
            
# 使用回溯的无剪枝暴力搜索
def backtrack_predict(sentence: str,
            predict_word: str,
            model,
            tokenizer,
            top_k=10,
            fast_mode=True,
            strict_mode=True):
```

> 由于BERT的自编码特性，导致其在预测MASK时，顺序不同会导致预测结果不同，如果启用`fast_mode`，则会正向和反向分别对输入进行预测，可以提升一点准确率(2%左右)，但是会带来更大的性能开销。

> `strict_mode`会对输入进行检查，以判断其是否为一个真实存在的汉语词汇。

### 如何微调模型

请参考[TrainExample.ipynb](https://github.com/IgarashiAkatuki/CNMBert/blob/main/TrainExample.ipynb),在数据集的格式上，只要保证csv的第一列为要训练的语料即可。

### Q&A

Q: 感觉这个东西准确度有点低啊

A: 可以尝试设置`fast_mode`和`strict_mode`为`False`。 模型是在很小的数据集(200w)上进行的预训练，所以泛化能力不足很正常，，，可以在更大数据集或者更加细分的领域进行微调，具体微调方式和[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)差别不大，只需要将`DataCollactor`替换为`CustomBertModel.py`中的`DataCollatorForMultiMask`。

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

