# # def profile_qa_agent(state, model):
# #     prompt = f"""
# # Answer using ONLY the paper profile.

# # Paper profile:
# # {state["paper_profile"]}

# # Question:
# # {state["user_query"]}
# # """

# #     answer = model.generate(prompt, max_new_tokens=200)

# #     return {
# #         **state,
# #         "final_answer": answer.strip()
# #     }
def profile_qa_agent(state, model):
    profile = state.get("paper_profile")

    if not profile:
        return {
            **state,
            "final_answer": "Paper profile not available."
        }

    prompt = f"""
Using ONLY the information below, answer the question.

Paper profile:
{profile}

Question:
{state["user_query"]}
"""

    answer = model.generate(prompt, max_new_tokens=200)

    return {
        **state,
        "final_answer": answer.strip()
    }
