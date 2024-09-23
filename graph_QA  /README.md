# Llamaindex graph rag
![image](https://github.com/user-attachments/assets/1890bb28-6374-4b01-a9df-8f32fb812877)

two main parts: extractor and retriever

## Extractor
https://docs.llamaindex.ai/en/stable/examples/property_graph/Dynamic_KG_Extraction/#3-schemallmpathextractor

![image](https://github.com/user-attachments/assets/6e0e0eac-0084-4194-a25f-b924a13105f6)

### ImplicitPathExtractor
link the previous and next sentence
![image](https://github.com/user-attachments/assets/d1100701-8100-4b48-8630-ea9439bb8569)


## SimpleLLMExtractor
Entity and relation recognition

```SimpleLLMPathExtractor: This extractor creates a basic knowledge graph without any predefined schema. It may produce a larger number of diverse relationships but might lack consistency in entity and relation naming.```
![image](https://github.com/user-attachments/assets/01c542e2-a034-4487-ae62-7177d64d2153)

## SchemaLLMPathExtractor
```SchemaLLMPathExtractor: With a predefined schema, this extractor produces a more structured graph. The entities and relations are limited to those specified in the schema, which can lead to a more consistent but potentially less comprehensive graph. Even if we set "strict" to false, the extracted KG Graph doesn't reflect the LLM's pursuit of trying to find new entities and types that fall outside of the input schema's scope.```
![image](https://github.com/user-attachments/assets/75fb2f03-a792-43ae-9f34-86627d7eb2d9)

## DynamicLLMPathExtractor
```The DynamicLLMPathExtractor graph should show a balance between diversity and consistency, potentially capturing important relationships that the schema-based approach might miss while still maintaining some structure.```

# Retrievers
![image](https://github.com/user-attachments/assets/0ab25e33-bac7-401d-b66b-1e9aef540cea)
