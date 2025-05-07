import os
from dotenv import load_dotenv
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
import getpass
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_perplexity import ChatPerplexity

load_dotenv()    

if "PPLX_API_KEY" not in os.environ:
    os.environ["PPLX_API_KEY"] = getpass.getpass("Enter your Perplexity API key: ")

index = LlamaCloudIndex(
  name="Dandesign-17159",
  project_name="Default",
  organization_id=os.getenv("LLAMA_INDEX_ORG_ID"),
  api_key=os.getenv("LLAMA_INDEX_KEY"),
)
# query = "What are the three stages of integration in multimodal-LLM?"
query = "What is Lady Gaga's real name?"

# configure retriever
retriever = index.as_retriever(
  dense_similarity_top_k=3,
  sparse_similarity_top_k=3,
  alpha=0.5,
  enable_reranking=True, 
  include_metadata=True
)

nodes = index.as_retriever().retrieve(query)
response = index.as_query_engine().query(query)

print("Nodes:")
for node in nodes:
    print(node)
    print(node.metadata.get("last_modified_at"))
    print("--------------------------------------")
    
print("Response:")
print(response) 

# 初始化 Perplexity 聊天模型
chat = ChatPerplexity(temperature=0.7, model="sonar-pro")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

chain = prompt | chat
response = chain.invoke({"input": "What is Trump's tariff policy?"})

# 输出回答
print("Response:")
print(response)

#Response:
#content="## Overview of Trump's Tariff Policy (2025)\n\nPresident Donald J. Trump’s 2025 tariff policy represents a significant escalation of protectionist trade measures compared to his first term. The latest executive orders and policy statements impose broad and targeted tariffs on U.S. imports, citing national security, economic competitiveness, and the need for reciprocity as primary justifications.\n\n**Key Elements of the 2025 Tariff Policy:**\n\n- **Universal Tariff:**  \n  As of April 2, 2025, President Trump ordered a minimum 10% tariff on all U.S. imports. This baseline tariff applies to imports from every country and took effect on April 5, 2025[1][3].\n\n- **Higher Tariffs on Specific Countries:**  \n  Imports from 57 targeted nations, including major trading partners, face higher tariffs, ranging from 11% to 50%. These elevated tariffs were implemented starting April 9, 2025[1][3].\n\n- **Special Tariffs on North American and Chinese Imports:**  \n  In February 2025, Trump imposed a 25% additional tariff on all products from Canada and Mexico, and a 10% additional tariff on imports from China. Canadian energy resources were subject to a lower 10% tariff[2][4].\n\n## Policy Rationale\n\n- **National Emergency Declaration:**  \n  Trump has invoked the International Emergency Economic Powers Act (IEEPA) to justify the tariffs, framing the U.S. trade deficit and non-reciprocal trade practices as a national emergency that threatens economic and national security[3].\n\n- **Targeting Drug Trafficking:**  \n  The tariffs on Canada, Mexico, and China are also justified as a response to the flow of illegal drugs—particularly fentanyl—into the United States, linking trade measures to the broader fight against narcotics and organized crime[2].\n\n- **Reciprocity and Economic Competitiveness:**  \n  The administration argues that persistent trade deficits and the lack of reciprocity in global trade harm American workers and industries, hollow out U.S. manufacturing, and increase dependency on foreign supply chains[3].\n\n## Economic Impact Projections\n\n- **Revenue Generation:**  \n  The tariffs are projected to raise $5.2 trillion in new revenue over the next decade, even after accounting for reduced demand for imports due to higher prices[1].\n\n- **Reduced Imports:**  \n  U.S. imports are expected to decrease by $6.9 trillion over the next ten years as a result of these tariffs[1].\n\n- **Potential for Federal Debt Reduction:**  \n  The revenue from tariffs could be used to reduce federal debt, according to projections, though lower trade volumes may also slow economic growth[1].\n\n## Duration and Flexibility\n\n- The tariffs will remain in effect until President Trump determines that the threats posed by trade deficits, nonreciprocal treatment, or drug trafficking have been resolved or sufficiently mitigated[3].\n\n## Summary Table: Main Tariff Rates (2025)\n\n| Country/Region         | Tariff Rate        | Rationale                          |\n|------------------------|--------------------|-------------------------------------|\n| All imports            | 10% minimum        | Economic/national security, reciprocity[1][3] |\n| 57 targeted countries  | 11%–50%            | Largest trade deficits, national emergency[1][3] |\n| Canada & Mexico        | 25%                | Illegal drug flow, national security[2][4] |\n| China                  | +10% (over baseline) | Drug precursors, trade practices[2][4] |\n| Canadian energy        | 10%                | Exemption for energy[2]             |\n\n## Conclusion\n\nTrump’s 2025 tariff policy is characterized by a sweeping 10% minimum tariff on all imports, significantly higher tariffs on select trading partners, and targeted measures against Canada, Mexico, and China. The policy aims to address trade imbalances, protect U.S. industries, and combat illegal drug trafficking, marking the most expansive use of tariffs by a U.S. president in recent history[1][2][3][4]." additional_kwargs={'citations': ['https://budgetmodel.wharton.upenn.edu/issues/2025/4/10/economic-effects-of-president-trumps-tariffs', 'https://www.whitehouse.gov/fact-sheets/2025/02/fact-sheet-president-donald-j-trump-imposes-tariffs-on-imports-from-canada-mexico-and-china/', 'https://www.whitehouse.gov/fact-sheets/2025/04/fact-sheet-president-donald-j-trump-declares-national-emergency-to-increase-our-competitive-edge-protect-our-sovereignty-and-strengthen-our-national-and-economic-security/', 'https://www.american.edu/sis/news/20241205-understanding-trump-tariffs-2-0.cfm', 'https://www.pbs.org/newshour/economy/a-timeline-of-trumps-tariff-actions-so-far']} response_metadata={'model_name': 'sonar-pro'} id='run--5fb98123-49a9-47c3-8a6e-16dfe0d3a544-0' usage_metadata={'input_tokens': 13, 'output_tokens': 839, 'total_tokens': 852}


# print("Response:")
# print(response.content)

# # 从 response_metadata 拿 citations
# print("\nCitations:")
# for cite in response.response_metadata.get("citations", []):
#     print(f"- {cite['title']} ({cite['published_at']}): {cite['url']}")