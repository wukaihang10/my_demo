# RAG 基础

RAG 的全称是 Retrieval-Augmented Generation，即检索增强生成。
它会先从外部知识库中找到和用户问题相关的资料，
再将资料提供给大语言模型生成答案。

# Agent 工具调用

Agent 可以通过工具访问模型参数之外的信息。
工具执行结果通常会作为 observation 返回给 Agent，
使 Agent 能够根据外部结果继续推理。

# 文本切分

长文档通常需要切成多个 Chunk。
Chunk 太大会导致语义不集中，Chunk 太小则可能缺少回答问题所需的上下文。