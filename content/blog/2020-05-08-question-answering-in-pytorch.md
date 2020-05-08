---
title: Question Answering in PyTorch
date: 2020-05-08T20:31:28.019Z
description: Build a Question Answering system using PyTorch
---
Hello and welcome to this tutorial where we will learn how to do question answering with PyTorch. For this, we will be using the [transformers](http://transformer.huggingface.co/) library. This is a library of pre-trained transformers that we can use for various tasks, like text summarization and question answering.

We will be using this library to do question answering. You will be surprised to see how well it does. Or at least I was surprised.

## What is question answering?

Before we get started, let me answer the question, What is question answering? Let's use an example to demonstrate. At the time of writing, there is a pandemic going on. It is the coronavirus pandemic, so our model will answer questions about the pandemic.

You have to give the model a paragraph of information and you can ask questions from that paragraph. So let's say that we have this paragraph of information:

> Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment. Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness. The best way to prevent and slow down transmission is be well informed about the COVID-19 virus, the disease it causes and how it spreads. Protect yourself and others from infection by washing your hands or using an alcohol based rub frequently and not touching your face. The COVID-19 virus spreads primarily through droplets of saliva or discharge from the nose when an infected person coughs or sneezes, so it’s important that you also practice respiratory etiquette (for example, by coughing into a flexed elbow). At this time, there are no specific vaccines or treatments for COVID-19. However, there are many ongoing clinical trials evaluating potential treatments. WHO will continue to provide updated information as soon as clinical findings become available.

We will ask the model the following question:

> What is COVID-19?

The model has to find a piece of text in the paragraph that answers this question. In this case, the answer would look like this:

> An infectious disease caused by a newly discovered coronavirus

Different models will give different answers. The answer above was given by a model called `BERT`. Here are answers from a few other models:

```
DistilBERT: Coronavirus disease 
Albert: coronavirus disease 
Roberta: an infectious disease caused by a newly discovered coronavirus
```

You can see that the`DistilBERT` and `Albert` give the same answer while `Roberta` gives the same answer as `BERT`.  Let's now dive in into question answering.

## Getting started

Make sure that you have installed the transformers library that I mentioned at the beginning. If not installed, run the following command

```sh
pip install transformers
```

You will also need to have PyTorch installed. We need to import our libraries now and for this, we only need 2 imports.

```python
import torch
import transformers
```

## Loading the model

We will now load the model in. Here, we are loading `BERT`:

```python
### Bert ###
tokenizer_bert = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = transformers.BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
```

This will take some time since the model and tokenizer are quite large. Once finished, you can continue on.

## Answering questions

The model will return the start and end position of the answer that it found in the paragraph. When we feed the question into the model, we have to do some preprocessing first, but that's why we loaded in the tokenizer. Below is the code for answering a question about a paragraph

```python
def  answer_simple(question, text, tokenizer, model):
	inputs = tokenizer.encode_plus(question, text, 
		add_special_tokens=True, 
		return_tensors="pt"
	)
	input_ids = inputs["input_ids"].tolist()[0]
	text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
	answer_start_scores, answer_end_scores = model(**inputs)
	answer_start = torch.argmax(answer_start_scores)
	answer_end = torch.argmax(answer_end_scores) + 1
	answer = tokenizer.convert_tokens_to_string(
		tokenizer.convert_ids_to_tokens(
			input_ids[answer_start:answer_end]
		)
	)
	return answer
```

Here is an example of the code running:

```python
>>> covid_paragraph = "Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment. Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness. The best way to prevent and slow down transmission is be well informed about the COVID-19 virus, the disease it causes and how it spreads. Protect yourself and others from infection by washing your hands or using an alcohol based rub frequently and not touching your face. The COVID-19 virus spreads primarily through droplets of saliva or discharge from the nose when an infected person coughs or sneezes, so it’s important that you also practice respiratory etiquette (for example, by coughing into a flexed elbow). At this time, there are no specific vaccines or treatments for COVID-19. However, there are many ongoing clinical trials evaluating potential treatments. WHO will continue to provide updated information as soon as clinical findings become available."
>>> print(answer_simple("What is COVID-19?", covid_paragraph, tokenizer_bert, model_bert))
an infectious disease caused by a newly discovered coronavirus
```

You can see that it answers the question correctly. This is all we have to do in order to do question answering in PyTorch.

> **Quick Note**: You have got to ask a good question to the model, otherwise it will perform bad. Also, the answer to your question should lie somewhere in the text that you have given.

## Conclusion

So, we've reached the end and now you know how to use a question answering model in PyTorch. But, there's more to this. There are other models and other ways. If you check out the bonus section, you will find how to load other models and easier ways to do question answering. So go ahead and check that out!

## Bonus

### Loading other models

There are other models that can answer question. The following code will show how to load 2 other models.

```python
### Distil Bert ###
tokenizer_distil_bert = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-cased')
model_distil_bert = transformers.DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')
### Albert ###
tokenizer_albert = transformers.AutoTokenizer.from_pretrained('ktrapeznikov/albert-xlarge-v2-squad-v2')
model_albert = transformers.AutoModelForQuestionAnswering.from_pretrained('ktrapeznikov/albert-xlarge-v2-squad-v2')
```

We loaded DistilBERT and Albert here. If you head over [Hugging Face's](https://huggingface.co/) site and select the `PyTorch` and `question-answering` tags in the model search, you will find more models there.

### Answering with a pipline

In this section, I will show you a simpler way to do question answering using the pipeline object that the transformers library gives. Here is how you do it.

We have to first create a pipeline,  and that is very simple to do.

```python
pipeline = transformers.pipeline("question-answering")
```

If you want to load in a custom model, then use this code:

```python
pipeline = transformers.pipeline('question-answering', model='distilbert-base-uncased-distilled-squad', tokenizer='bert-base-uncased')
```

This loads in DistilBERT and both pipelines can be used for question answering.

Now we can answer questions. Here is a helper function that will make the job easier:

```python
answer_pipeline = lambda question, text, pipeline: pipeline(question=question, context=text)["answer"]
```

The pipeline also returns a lot of other information, and this function just extracts the answer. As you can see, answering with a pipeline is so simple that I was able to squeeze it into a lambda function. Using this function, we get something like this:

```python
>>> # Using the previous covid_paragraph
>>> print(answer_pipeline("What is COVID-19?", covid_paragraph, pipeline))
an infectious disease caused by a newly discovered coronavirus.
```

This returns the same answer as before, and we see that it works. I found that it took a little longer to run but it worked.

Anyways, that's it for this post and I will see you next time. Oh, and there are no more bonus sections. The next post will be about text generation.
