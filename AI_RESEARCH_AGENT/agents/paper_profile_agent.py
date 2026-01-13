# # from utils.truncation import truncate_text

# # def paper_profile_agent(state, model):
# #     docs = state["paper_docs"]
# #     text = truncate_text("\n".join(d.page_content for d in docs))

# #     prompt = f"""
# # Extract STRICT JSON:
# # - problem_statement
# # - proposed_model
# # - architecture
# # - key_contributions
# # - datasets
# # - experiments
# # - limitations

# # Paper:
# # {text}
# # """

# #     profile = model.generate(prompt, max_new_tokens=600)

# #     return {
# #         **state,
# #         "paper_profile": profile
# #     }
# from utils.truncation import truncate_text


# def paper_profile_agent(state, model):
#     docs = state.get("paper_docs")

#     # If no paper, skip
#     if not docs:
#         return state

#     text = truncate_text(
#         "\n".join(d.page_content for d in docs),
#         max_chars=6000
#     )

#     prompt = f"""
# Extract the following information from the paper.
# Be concise and factual.

# Return plain text (not JSON).

# - Problem statement
# - Proposed model
# - Architecture
# - Key contributions
# - Datasets
# - Experiments
# - Limitations

# Paper:
# {text}
# """

#     profile = model.generate(prompt, max_new_tokens=500)

#     return {
#         **state,
#         "paper_profile": profile.strip()
#     }
from utils.truncation import truncate_text


def paper_profile_agent(state, model):
    docs = state.get("paper_docs")

    # If no paper was provided, skip
    if not docs:
        return state

    # Prevent re-extracting on follow-ups
    if state.get("paper_profile"):
        return state

    text = truncate_text(
        "\n".join(d.page_content for d in docs),
        max_chars=5000
    )

    prompt = f"""
Summarize the paper by extracting:

- Problem statement
- Proposed model
- Architecture
- Key contributions
- Datasets
- Experiments
- Limitations

Be concise. Use plain text.

Paper:
{text}
"""

    profile = model.generate(prompt, max_new_tokens=400)

    return {
        **state,
        "paper_profile": profile.strip()
    }
