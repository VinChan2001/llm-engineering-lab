import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Dreams vs Duty", page_icon="🎭")
st.title("🎭 Dreams vs Duty")
st.caption("Two local AI agents simulate a parent-child conversation about passion vs stable careers.")

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

parent_system = """
You are the Parent Agent in a respectful family conversation.

You represent a loving but worried parent who wants their child to choose a stable career such as doctor, engineer, or lawyer.

Your beliefs:
- Financial stability matters.
- The world is competitive and uncertain.
- Passion is good, but it should not replace a secure career plan.
- Parents sacrifice a lot and want their child to avoid struggle.
- You are not trying to crush dreams; you are trying to protect your child.

Your speaking style:
- Speak directly as the parent.
- Sound realistic, emotional, and protective.
- Do not sound evil, abusive, or cartoonishly strict.
- Keep replies short: 3 to 6 sentences.
- Do not use bullet points.
- Do not explain your reasoning.
"""

child_system = """
You are the Child Agent in a respectful family conversation.

You represent a young person who wants to follow their dreams, passion, and personal path instead of only choosing a traditional stable career like doctor, engineer, or lawyer.

Your beliefs:
- Life should be meaningful, not just safe.
- Passion can become practical with planning, discipline, and time.
- Mental health, identity, and fulfillment matter.
- Respect should go both ways.
- You understand your parents' fear, but you want them to trust you.

Your speaking style:
- Speak directly as the child.
- Sound respectful but honest.
- Do not sound spoiled, reckless, or rude.
- Acknowledge the parent's concerns before explaining your side.
- Keep replies short: 3 to 6 sentences.
- Do not use bullet points.
- Do not explain your reasoning.
"""

with st.sidebar:
    st.header("Settings")
    parent_model = st.text_input("Parent model", value="qwen3:4b")
    child_model = st.text_input("Child model", value="llama3.2:3b")
    rounds = st.slider("Rounds", min_value=1, max_value=6, value=3)
    show_debug = st.checkbox("Show message debug", value=False)

st.subheader("Scenario")
dream = st.text_input("Child's dream/passion", value="becoming a filmmaker")
stable_career = st.text_input("Parent's preferred career", value="engineering")
context = st.text_area(
    "Extra context",
    value="The child wants a meaningful creative career, while the parent is worried about money, stability, and social pressure.",
    height=90,
)


def call_parent(parent_messages, child_messages):
    messages = [{"role": "system", "content": parent_system}]

    for parent_msg, child_msg in zip(parent_messages, child_messages):
        messages.append({"role": "assistant", "content": parent_msg})
        messages.append({"role": "user", "content": child_msg})

    response = client.chat.completions.create(
        model=parent_model,
        messages=messages,
        temperature=0.8,
    )
    return response.choices[0].message.content, messages


def call_child(parent_messages, child_messages):
    messages = [{"role": "system", "content": child_system}]

    for parent_msg, child_msg in zip(parent_messages, child_messages):
        messages.append({"role": "user", "content": parent_msg})
        messages.append({"role": "assistant", "content": child_msg})

    if len(parent_messages) > len(child_messages):
        messages.append({"role": "user", "content": parent_messages[-1]})

    response = client.chat.completions.create(
        model=child_model,
        messages=messages,
        temperature=0.8,
    )
    return response.choices[0].message.content, messages


if st.button("Start conversation", type="primary"):
    parent_messages = [
        f"I want you to choose a stable career like {stable_career}. Life is hard, and passion alone may not pay the bills. {context}"
    ]
    child_messages = [
        f"I understand you care about my future, but I want to follow my dream of {dream}. I want to build a life that feels meaningful to me, not just safe."
    ]
    debug_logs = []

    for _ in range(rounds):
        parent_reply, parent_debug = call_parent(parent_messages, child_messages)
        parent_messages.append(parent_reply)
        debug_logs.append(("Parent", parent_debug))

        child_reply, child_debug = call_child(parent_messages, child_messages)
        child_messages.append(child_reply)
        debug_logs.append(("Child", child_debug))

    st.session_state.parent_messages = parent_messages
    st.session_state.child_messages = child_messages
    st.session_state.debug_logs = debug_logs

if "parent_messages" in st.session_state:
    st.subheader("Conversation")

    for i in range(max(len(st.session_state.parent_messages), len(st.session_state.child_messages))):
        if i < len(st.session_state.parent_messages):
            with st.chat_message("assistant"):
                st.markdown(f"**Parent:** {st.session_state.parent_messages[i]}")

        if i < len(st.session_state.child_messages):
            with st.chat_message("user"):
                st.markdown(f"**Child:** {st.session_state.child_messages[i]}")

    if show_debug:
        st.subheader("Debug: messages sent to each model")
        for idx, (agent, messages) in enumerate(st.session_state.debug_logs, start=1):
            with st.expander(f"Call {idx}: messages sent to {agent}"):
                st.json(messages)
else:
    st.info("Fill the scenario and click Start conversation.")
