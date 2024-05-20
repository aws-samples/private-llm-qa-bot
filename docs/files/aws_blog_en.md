## Amazon AWS Official Blog

# Fine-Tuning GPT-J Using an Amazon SageMaker Hugging Face Estimator and the Model Parallel Library

In this post, we will introduce a guide and best practices for training large language models (LLMs) using Amazon SageMaker's distributed model parallel library to reduce training time and cost. You will learn how to easily train a 6 billion parameter GPT-J model on SageMaker. Finally, we will share the key capabilities of SageMaker's distributed model parallelism that help speed up training time.

## Transformer Neural Networks

Transformer neural networks are a popular deep learning architecture for tackling sequence-to-sequence tasks. The architecture uses attention as the learning mechanism to achieve near human-level performance. Some other useful characteristics of this architecture compared to earlier generations of natural language processing (NLP) models include the distributed, scalable, and pre-trained abilities. Transformer-based models can be applied to different use cases when working with text data such as search, chatbots, and more. Transformers leverage the concept of pre-training to acquire intelligence from large datasets. The pre-trained transformer can be used as-is or fine-tuned on datasets that can be much smaller and specific to your business.

## Hugging Face on SageMaker

Hugging Face is a company that develops some of the most popular open-source libraries providing state-of-the-art NLP technology based on the transformer architecture. The Hugging Face Transformer, Tokenizer, and Datasets libraries provide APIs and tools to download and make predictions using pre-trained models in multiple languages. SageMaker allows you to use Hugging Face models directly from the Hugging Face Model Hub for training, fine-tuning, and running inferences using the Hugging Face Estimator in the SageMaker SDK. This integration makes it easier to customize Hugging Face models for use cases in specific domains. Behind the scenes, the SageMaker SDK uses AWS Deep Learning Containers (DLCs), which are a set of pre-built Docker images for training and serving models provided by SageMaker. The DLCs are co-developed by AWS and Hugging Face. This integration also provides integration between the Hugging Face Transformer SDK and SageMaker's distributed training libraries, enabling you to scale training jobs across GPU clusters.

## Overview of the SageMaker Model Parallel Library

Model parallelism is a distributed training strategy that partitions the deep learning model across multiple devices within or across instances. Deep learning (DL) models with more layers and parameters perform better on complex tasks such as computer vision and NLP. However, the maximum model size that can be stored in a single GPU's memory is limited. During DL model training, the GPU memory constraint may become a bottleneck in the following ways:

SageMaker includes a distributed model parallel library that helps efficiently distribute and train DL models across multiple compute nodes, overcoming the limitations of training models on a single GPU. Additionally, the library also allows you to leverage EFA-enabled devices for optimal distributed training by providing low-latency, high-throughput, and OS bypass for inter-node communication performance.

Since large models like GPT-J have billions of parameters that exceed the GPU memory capacity of a single chip, they need to be partitioned across multiple GPUs. The SageMaker Model Parallel (SMP) library supports automatically partitioning the model across multiple GPUs. Using SageMaker's model parallel capability, SageMaker runs an initial profiling job on your behalf to analyze the model's compute and memory requirements. This information is then used to decide how to partition the model across GPUs to best optimize for targets such as maximizing speed or minimizing memory usage.

The capability also supports optional pipeline scheduling to maximize the overall utilization of available GPUs. The forward propagation of activations during the forward pass and the backpropagation of gradients during the backward pass need to be computed sequentially, limiting GPU utilization. SageMaker overcomes the sequential computation limitation using pipeline scheduling by splitting mini-batches into micro-batches that can be processed in parallel across different GPUs. The SageMaker Model Parallel supports two modes of pipeline parallelism:

### Tensor Parallelism

Individual layers or nn.Modules are split across devices using tensor parallelism to run concurrently. The following is a simple example illustrating how the library partitions a model into four layers to achieve bidirectional tensor parallelism ("tensor_parallel_degree": 2). The layers of each model replica are split in half and distributed across two GPUs. In this example, the data parallelism degree is eight as the model parallel configuration also includes "pipeline_parallel_degree": 1 and "ddp": True. The library manages communication across distributed tensor model replicas.

![Tensor Parallelism](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2023/06/05/ML-8585-tensorp.jpg)

The benefit of this capability is that you can choose which layers or subset of layers to apply tensor parallelism to. To learn more about tensor parallelism in PyTorch and other memory-saving capabilities, and how to set up a combination of pipeline and tensor parallelism, refer to the SageMaker Model Parallel library's advanced features for PyTorch.

### SageMaker Sharded Data Parallelism

Sharded data parallelism is a memory-saving distributed training technique that splits the model's training state (model parameters, gradients, and optimizer states) across multiple GPUs within a data parallel group.

When scaling a training job to a large GPU cluster, sharding the training state across multiple GPUs can reduce the per-GPU memory footprint of the model. This has two benefits: it can accommodate larger models that would otherwise exhaust memory with standard data parallelism, and it can leverage the freed GPU memory to increase batch size.

Standard data parallelism replicates the training state across GPUs within a data parallel group and performs gradient aggregation through AllReduce operations. Essentially, sharded data parallelism trades off communication overhead for GPU memory efficiency. Using sharded data parallelism incurs higher communication cost, but each GPU's memory footprint (excluding memory usage from activations) is divided by the sharded data parallelism degree, allowing the GPU cluster to fit larger models.

SageMaker implements sharded data parallelism through MiCS. For more information, refer to Near-linear Scaling of Gigantic Model Training on AWS.

For more detailed information on how to apply sharded data parallelism in your training job, refer to Sharded Data Parallelism.

### Using the SageMaker Model Parallel Library

The SageMaker Model Parallel library comes bundled with the SageMaker Python SDK. You need to have the SageMaker Python SDK installed to use this library, which is pre-installed on the SageMaker notebook kernels. To leverage the capabilities of the SMP library in your PyTorch training script, the following changes are needed:

Refer to the following code snippet:

The SageMaker Model Parallel library's tensor parallelism provides out-of-the-box support for the following Hugging Face Transformer models:

## Best Practices for Performance Tuning with the SMP Library

When training large models, consider the following steps to enable the model to run with a reasonable batch size within GPU memory:

We conducted multiple experiments on SageMaker leveraging the SMP library to optimize the training and fine-tuning of GPT-J. We successfully reduced the training time of GPT-J on SageMaker from 58 minutes down to under 10 minutes—a six-fold reduction in single training time. The time for initialization, downloading the model and dataset from Amazon Simple Storage Service (Amazon S3) took less than 1 minute, the time for profiling and automatic partitioning using GPU as the tracing device took less than 1 minute, and the training time using tensor parallelism, FP16 precision, and the SageMaker Hugging Face Estimator on a single ml.p4d.24xlarge instance was 8 minutes.

As a best practice, when training GPT-J on SageMaker to reduce training time, our recommendations are as follows:

## Leveraging the SMP Library to Train and Fine-Tune the GPT-J Model on SageMaker

The Amazon SageMaker Examples public repository provides a hands-on, step-by-step code example. Navigate to the training/distributed_training/pytorch/model_parallel/gpt-j folder. Select the gpt-j folder and open the train_gptj_smp_tensor_parallel_notebook.jpynb Jupyter notebook (for the tensor parallel example) and train_gptj_smp_notebook.ipynb (for the pipeline parallel example). You can find a code walkthrough in our Generative AI on Amazon SageMaker workshop.  

This notebook will walk you through how to use the tensor parallelism capability provided by the SageMaker Model Parallel library. You will learn how to perform FP16 training of the GPT-J model using tensor parallelism and pipeline parallelism on the GLUE sst2 dataset.

## Summary

The SageMaker Model Parallel library provides several capabilities. You can reduce cost and speed up training LLMs on SageMaker. You can also learn and run sample code for BERT, GPT-2, and GPT-J in the Amazon SageMaker Examples public repository. To learn more about AWS best practices for training LLMs using the SMP library, refer to the following resources:

To learn how one of our customers achieved low-latency GPT-J inference on SageMaker, see How Mantium Leveraged DeepSpeed for Low-Latency GPT-J Inference on Amazon SageMaker.

If you're looking to reduce time to market and costs for your LLMs, SageMaker can help. Let us know what you build!

Original URL: https://aws.amazon.com/blogs/machine-learning/fine-tune-gpt-j-using-an-amazon-sagemaker-hugging-face-estimator-and-the-model-parallel-library/

## About the Authors

Dr. Zmnako Awrahman is a Practice Manager, Machine Learning Specialist, and a member of the ML Tech Field Community (TFC) at Amazon Web Services Global Capability Center. He helps customers harness the power of cloud to extract value from their data through data analytics and machine learning.  

![Zmnako Awrahman](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2023/06/05/ML-8585-zmnaka-1.png)

Roop Bains is a Sr. Machine Learning Solutions Architect at AWS. He is passionate about helping customers innovate and achieve their business goals using AI and ML. He assists customers in training, optimizing, and deploying deep learning models.

![Roop Bains](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2023/06/05/ML-8585-roopbain.png)

Anastasia Pachni Tsitiridou is a Solutions Architect at AWS. Anastasia is based in Amsterdam, supporting the cloud journey of software enterprises in the Benelux economic union. Before joining AWS, she studied Electrical and Computer Engineering with a major in Computer Vision. Nowadays, her favorite is working with very large language models.

![Anastasia Pachni Tsitiridou](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2023/06/05/ML-8585-anapt.png)

Dhawal Patel is a Principal Machine Learning Architect at AWS. He has worked across organizations ranging from large enterprises to mid-sized startups, solving problems related to distributed computing and AI. His focus is on deep learning including NLP and computer vision domains. He helps customers achieve high-performance model inference on SageMaker.

![Dhawal Patel](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2023/06/05/ML-8585-dhawalkp.png)

Wioletta Stobieniecka is a Data Scientist at AWS Professional Services. Throughout her career, she has delivered multiple analytics-driven projects across different industries, including banking, insurance, telecommunications, and public sector. Her expertise in advanced statistical methods and machine learning is combined with a sharp business acumen. She brings the latest AI technologies to create value for customers.

![Wioletta Stobieniecka](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2023/06/05/ML-8585-splwis.png)

Rahul Huilgol is a Senior Software Development Engineer in the Distributed Deep Learning domain at Amazon Web Services. 

![Rahul Huilgol](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2023/06/05/ML-8585-huilgolr.png)