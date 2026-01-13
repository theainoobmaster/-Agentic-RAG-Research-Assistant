from retriever.persistent_store import load_or_create_store
from utils.truncation import truncate_text

def react_paper_agent(state, model):
    store = load_or_create_store()
    retriever = store.as_retriever(search_kwargs={"k": 4})

    prompt = f"""
Use:
- Action: search[query]
- Action: finish[answer]

Question:
{state["user_query"]}
"""

    for _ in range(3):
        response = model.generate(prompt, max_new_tokens=256)

        if "search[" in response:
            q = response.split("search[",1)[1].split("]",1)[0]
            docs = retriever.get_relevant_documents(q)
            context = truncate_text(
                "\n".join(d.page_content for d in docs)
            )
            prompt += f"\nObservation:\n{context}\n"

        elif "finish[" in response:
            answer = response.split("finish[",1)[1].rsplit("]",1)[0]
            return {**state, "final_answer": answer.strip()}

    return {**state, "final_answer": "Answer not found in the paper."}
