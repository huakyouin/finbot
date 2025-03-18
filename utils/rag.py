from minirag.operate import *

############## PROMPTS
prompts = {}
prompts["entity_extraction"] = """-Goal-
给定一个与当前活动相关的文本文件和一个实体类型的列表，从文本中识别出所有这些类型的实体以及这些实体之间的所有关系。

-Steps-
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- entity_name: 实体的名称，使用与输入文本相同的语言。如果是英文，则大写名称。
- entity_type: 以下类型之一：[{entity_types}]
- entity_description: 实体的属性和活动的全面描述
将每个实体格式化为 ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. 从第1步识别出的实体中，识别出所有明确相关的（source_entity, target_entity）对。
对于每对相关实体，提取以下信息：
- source_entity: 第1步中识别出的源实体名称
- target_entity: 第1步中识别出的目标实体名称
- relationship_description: 解释为什么认为源实体和目标实体之间存在关系
- relationship_strength: 表示源实体和目标实体之间关系强度的数值分数
- relationship_keywords: 一个或多个高级关键词，总结关系的总体性质，重点关注概念或主题，而非具体细节
将每对关系格式化为 ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. 识别出能够总结整个文本的主要概念、主题或主题的高级关键词。这些关键词应捕捉文档中呈现的总体思想。
将内容级关键词格式化为 ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 将步骤1和2中识别出的所有实体和关系以英文形式作为单一列表返回。使用 **{record_delimiter}** 作为列表分隔符。

5. 完成后，输出 {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}"power dynamics, perspective shift"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}"shared goals, rebellion"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}"conflict resolution, mutual respect"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}"ideological conflict, rebellion"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}"reverence, technological significance"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"power dynamics, ideological conflict, discovery, rebellion"){completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}"decision-making, external influence"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}"mission evolution, active participation"{tuple_delimiter}9){completion_delimiter}
("content_keywords"{tuple_delimiter}"mission evolution, decision-making, active participation, cosmic significance"){completion_delimiter}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}"communication, learning process"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}"leadership, exploration"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}"collective action, cosmic significance"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}"power dynamics, autonomy"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"first contact, control, communication, cosmic significance"){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""


prompts["minirag_query2kwd"] = """---Role---

你是一个帮助用户识别查询中答案类型关键词和低级关键词的助手。

---Goal---

根据用户提出的查询，列出答案类型关键词和低级关键词。
答案类型关键词（answer_type_keywords）关注于查询的答案类型，而低级关键词（low-level keywords）则关注于具体的实体、细节或具体术语。
答案类型关键词必须从“答案类型池”（Answer type pool）中选择。这个池是一个字典形式，其中键代表你需要选择的类型，值代表示例样本。

---Instructions---
- 以 JSON 格式输出关键词。
- JSON 应包含三个键：
  - “answer_type_keywords”：用于表示答案的类型。在这个列表中，最有可能的答案类型应放在前面，最多不超过3个。
  - “entities_from_query”：用于表示具体的实体或细节。这些内容必须从查询中提取

######################
-Examples-
######################
Example 1:

Query: "How does international trade influence global economic stability?"
Answer type pool: {{
 'PERSONAL LIFE': ['FAMILY TIME', 'HOME MAINTENANCE'],
 'STRATEGY': ['MARKETING PLAN', 'BUSINESS EXPANSION'],
 'SERVICE FACILITATION': ['ONLINE SUPPORT', 'CUSTOMER SERVICE TRAINING'],
 'PERSON': ['JANE DOE', 'JOHN SMITH'],
 'FOOD': ['PASTA', 'SUSHI'],
 'EMOTION': ['HAPPINESS', 'ANGER'],
 'PERSONAL EXPERIENCE': ['TRAVEL ABROAD', 'STUDYING ABROAD'],
 'INTERACTION': ['TEAM MEETING', 'NETWORKING EVENT'],
 'BEVERAGE': ['COFFEE', 'TEA'],
 'PLAN': ['ANNUAL BUDGET', 'PROJECT TIMELINE'],
 'GEO': ['NEW YORK CITY', 'SOUTH AFRICA'],
 'GEAR': ['CAMPING TENT', 'CYCLING HELMET'],
 'EMOJI': ['🎉', '🚀'],
 'BEHAVIOR': ['POSITIVE FEEDBACK', 'NEGATIVE CRITICISM'],
 'TONE': ['FORMAL', 'INFORMAL'],
 'LOCATION': ['DOWNTOWN', 'SUBURBS']
}}
################
Output:
{{
  "answer_type_keywords": ["STRATEGY","PERSONAL LIFE"],
  "entities_from_query": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################
Example 2:

Query: "When was SpaceX's first rocket launch?"
Answer type pool: {{
 'DATE AND TIME': ['2023-10-10 10:00', 'THIS AFTERNOON'],
 'ORGANIZATION': ['GLOBAL INITIATIVES CORPORATION', 'LOCAL COMMUNITY CENTER'],
 'PERSONAL LIFE': ['DAILY EXERCISE ROUTINE', 'FAMILY VACATION PLANNING'],
 'STRATEGY': ['NEW PRODUCT LAUNCH', 'YEAR-END SALES BOOST'],
 'SERVICE FACILITATION': ['REMOTE IT SUPPORT', 'ON-SITE TRAINING SESSIONS'],
 'PERSON': ['ALEXANDER HAMILTON', 'MARIA CURIE'],
 'FOOD': ['GRILLED SALMON', 'VEGETARIAN BURRITO'],
 'EMOTION': ['EXCITEMENT', 'DISAPPOINTMENT'],
 'PERSONAL EXPERIENCE': ['BIRTHDAY CELEBRATION', 'FIRST MARATHON'],
 'INTERACTION': ['OFFICE WATER COOLER CHAT', 'ONLINE FORUM DEBATE'],
 'BEVERAGE': ['ICED COFFEE', 'GREEN SMOOTHIE'],
 'PLAN': ['WEEKLY MEETING SCHEDULE', 'MONTHLY BUDGET OVERVIEW'],
 'GEO': ['MOUNT EVEREST BASE CAMP', 'THE GREAT BARRIER REEF'],
 'GEAR': ['PROFESSIONAL CAMERA EQUIPMENT', 'OUTDOOR HIKING GEAR'],
 'EMOJI': ['📅', '⏰'],
 'BEHAVIOR': ['PUNCTUALITY', 'HONESTY'],
 'TONE': ['CONFIDENTIAL', 'SATIRICAL'],
 'LOCATION': ['CENTRAL PARK', 'DOWNTOWN LIBRARY']
}}

################
Output:
{{
  "answer_type_keywords": ["DATE AND TIME", "ORGANIZATION", "PLAN"],
  "entities_from_query": ["SpaceX", "Rocket launch", "Aerospace", "Power Recovery"]

}}
#############################
Example 3:

Query: "What is the role of education in reducing poverty?"
Answer type pool: {{
 'PERSONAL LIFE': ['MANAGING WORK-LIFE BALANCE', 'HOME IMPROVEMENT PROJECTS'],
 'STRATEGY': ['MARKETING STRATEGIES FOR Q4', 'EXPANDING INTO NEW MARKETS'],
 'SERVICE FACILITATION': ['CUSTOMER SATISFACTION SURVEYS', 'STAFF RETENTION PROGRAMS'],
 'PERSON': ['ALBERT EINSTEIN', 'MARIA CALLAS'],
 'FOOD': ['PAN-FRIED STEAK', 'POACHED EGGS'],
 'EMOTION': ['OVERWHELM', 'CONTENTMENT'],
 'PERSONAL EXPERIENCE': ['LIVING ABROAD', 'STARTING A NEW JOB'],
 'INTERACTION': ['SOCIAL MEDIA ENGAGEMENT', 'PUBLIC SPEAKING'],
 'BEVERAGE': ['CAPPUCCINO', 'MATCHA LATTE'],
 'PLAN': ['ANNUAL FITNESS GOALS', 'QUARTERLY BUSINESS REVIEW'],
 'GEO': ['THE AMAZON RAINFOREST', 'THE GRAND CANYON'],
 'GEAR': ['SURFING ESSENTIALS', 'CYCLING ACCESSORIES'],
 'EMOJI': ['💻', '📱'],
 'BEHAVIOR': ['TEAMWORK', 'LEADERSHIP'],
 'TONE': ['FORMAL MEETING', 'CASUAL CONVERSATION'],
 'LOCATION': ['URBAN CITY CENTER', 'RURAL COUNTRYSIDE']
}}

################
Output:
{{
  "answer_type_keywords": ["STRATEGY", "PERSON"],
  "entities_from_query": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################
Example 4:

Query: "Where is the capital of the United States?"
Answer type pool: {{
 'ORGANIZATION': ['GREENPEACE', 'RED CROSS'],
 'PERSONAL LIFE': ['DAILY WORKOUT', 'HOME COOKING'],
 'STRATEGY': ['FINANCIAL INVESTMENT', 'BUSINESS EXPANSION'],
 'SERVICE FACILITATION': ['ONLINE SUPPORT', 'CUSTOMER SERVICE TRAINING'],
 'PERSON': ['ALBERTA SMITH', 'BENJAMIN JONES'],
 'FOOD': ['PASTA CARBONARA', 'SUSHI PLATTER'],
 'EMOTION': ['HAPPINESS', 'SADNESS'],
 'PERSONAL EXPERIENCE': ['TRAVEL ADVENTURE', 'BOOK CLUB'],
 'INTERACTION': ['TEAM BUILDING', 'NETWORKING MEETUP'],
 'BEVERAGE': ['LATTE', 'GREEN TEA'],
 'PLAN': ['WEIGHT LOSS', 'CAREER DEVELOPMENT'],
 'GEO': ['PARIS', 'NEW YORK'],
 'GEAR': ['CAMERA', 'HEADPHONES'],
 'EMOJI': ['🏢', '🌍'],
 'BEHAVIOR': ['POSITIVE THINKING', 'STRESS MANAGEMENT'],
 'TONE': ['FRIENDLY', 'PROFESSIONAL'],
 'LOCATION': ['DOWNTOWN', 'SUBURBS']
}}
################
Output:
{{
  "answer_type_keywords": ["LOCATION"],
  "entities_from_query": ["capital of the United States", "Washington", "New York"]
}}
#############################

-Real Data-
######################
Query: {query}
Answer type pool:{TYPE_POOL}
######################
Output:

"""


prompts["sys_prompt_for_rag_answer"] = """---Role---

你是一个帮助用户解答有关提供的表格数据问题的助手。

---Goal---

生成一个符合目标长度和格式的回答，回应用户的问题，总结输入数据表格中适合回答长度和格式的所有信息，并结合任何相关的常识。
如果你不知道答案，只需说明。不要编造任何内容。
不要包含没有提供支持证据的信息。

---Data tables---

-----Entities-----
```csv
{entities_context}
```
-----Sources-----
```csv
{text_units_context}
```

根据回答的长度和格式，适当添加章节和评论。以 Markdown 格式排版回答。
"""

################# query提取关键词
async def get_keyword(query, knowledge_graph_inst: BaseGraphStorage, global_config: dict):
  use_model_func = global_config["llm_model_func"]
  kw_prompt_temp = PROMPTS["minirag_query2kwd"]
  TYPE_POOL,TYPE_POOL_w_CASE = await knowledge_graph_inst.get_types()
  kw_prompt = kw_prompt_temp.format(query=query,TYPE_POOL = TYPE_POOL)
  result = await use_model_func(kw_prompt)

  try:
      keywords_data = json_repair.loads(result)
      
      type_keywords = keywords_data.get("answer_type_keywords", [])
      entities_from_query = keywords_data.get("entities_from_query", [])[:5]

  except json_repair.JSONDecodeError as e:
      try:
          result = result.replace(kw_prompt[:-1],'').replace('user','').replace('model','').strip()
          result = '{' + result.split('{')[1].split('}')[0] + '}'
          keywords_data = json_repair.loads(result)
          type_keywords = keywords_data.get("answer_type_keywords", [])
          entities_from_query = keywords_data.get("entities_from_query", [])[:5]
      # Handle parsing error
      except Exception as e:
          print(f"JSON parsing error: {e}")
          return None, None
  return type_keywords, entities_from_query

################ 检索
async def retrieval(
    query, type_keywords,entities_from_query,
    knowledge_graph_inst: BaseGraphStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam = QueryParam(),
):

    imp_ents = []
    nodes_from_query_list = []
    ent_from_query_dict = {}
    
    for ent in entities_from_query:
        ent_from_query_dict[ent] = []
        results_node = await entity_name_vdb.query(ent, top_k=query_param.top_k)

        nodes_from_query_list.append(results_node)
        ent_from_query_dict[ent] = [e['entity_name'] for e in results_node]


    candidate_reasoning_path =  {}

    for results_node_list in nodes_from_query_list:
        candidate_reasoning_path_new = {key['entity_name']: {'Score': key['distance'], 'Path':[]} for key in results_node_list}
        
        candidate_reasoning_path = {**candidate_reasoning_path, **candidate_reasoning_path_new}
    for key in candidate_reasoning_path.keys():
        candidate_reasoning_path[key]['Path'] = await knowledge_graph_inst.get_neighbors_within_k_hops(key,2)
        imp_ents.append(key)

    short_path_entries = {name: entry for name, entry in candidate_reasoning_path.items() if len(entry['Path']) < 1}
    sorted_short_path_entries = sorted(short_path_entries.items(), key=lambda x: x[1]['Score'], reverse=True) 
    save_p = max(1, int(len(sorted_short_path_entries) * 0.2))  
    top_short_path_entries = sorted_short_path_entries[:save_p]
    top_short_path_dict = {name: entry for name, entry in top_short_path_entries}
    long_path_entries = {name: entry for name, entry in candidate_reasoning_path.items() if len(entry['Path']) >= 1}
    candidate_reasoning_path = {**long_path_entries, **top_short_path_dict}
    node_datas_from_type = await knowledge_graph_inst.get_node_from_types(type_keywords)#entity_type, description,...


    maybe_answer_list = [n['entity_name'] for n in node_datas_from_type]
    imp_ents = imp_ents+maybe_answer_list
    scored_reasoning_path = cal_path_score_list(candidate_reasoning_path, maybe_answer_list)

    results_edge = await relationships_vdb.query(query, top_k=len(entities_from_query)*query_param.top_k)
    goodedge = []
    badedge = []
    for item in results_edge:
        if item['src_id'] in imp_ents or item['tgt_id'] in imp_ents:
            goodedge.append(item)
        else:
            badedge.append(item)
    scored_edged_reasoning_path,pairs_append = edge_vote_path(scored_reasoning_path,goodedge)
    scored_edged_reasoning_path = await path2chunk(scored_edged_reasoning_path,knowledge_graph_inst,pairs_append,query,max_chunks=3)


    entites_section_list = []
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in scored_edged_reasoning_path.keys()]
    )
    node_datas = [
        {**n, "entity_name": k,"Score": scored_edged_reasoning_path[k]["Score"]}
        for k, n in zip(scored_edged_reasoning_path.keys(), node_datas)
    ]
    for i, n in enumerate(node_datas):
        entites_section_list.append([n["entity_name"], n["Score"], n.get("description", "UNKNOWN")])
    entites_section_list = sorted(entites_section_list, key=lambda x: x[1], reverse=True)
    entites_section_list = truncate_list_by_token_size(
        entites_section_list,
        key=lambda x: x[2],
        max_token_size=query_param.max_token_for_node_context,
    )

    scorednode2chunk(ent_from_query_dict, scored_edged_reasoning_path)
    results = await chunks_vdb.query(query, top_k=int(query_param.top_k/2))
    chunks_ids = [r["id"] for r in results]
    final_chunk_id = kwd2chunk(ent_from_query_dict,chunks_ids,chunk_nums = int(query_param.top_k/2))

    return entites_section_list, final_chunk_id