**Text Classification with 🤗 Transformers**


In this example , the 🤗 [Transformers](https://huggingface.co/transformers/index.html) library is used to classify tweets as Hate speech,Offensive language or neither.

The library provides general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between Jax, PyTorch and TensorFlow.

The approach followed was to fine-tune a pre-trained BERT model. 

The dataset used was the [hate speech and offensive language dataset](https://www.kaggle.com/mrmorj/hate-speech-and-offensive-language-dataset)