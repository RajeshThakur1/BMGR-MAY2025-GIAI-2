import autogen
from dotenv import load_dotenv
import os
load_dotenv()
import streamlit as st
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ITSupportBot(autogen.AssistantAgent):
    def __init__(self, name: str, memory=None, model: str="gpt-4o-mini"):
        super().__init__(name=name)
        self.memory = memory if memory is not None else {}
        self.model = model

    def _get_gpt_response(self, message, context):
        """
        Step 4: Calls OpenAI's GPT model to generate a response with past issue history.
        - Constructs a prompt including past conversation history
        - Sends the prompt to GPT-4o-mini
        - Returns the generated response
        
        """
        prompt = f"User's previous issue: {context}\nNew issue: {message}\nIT Support Response:"

        response = client.chat.completions.create(
            model = self.model,
            messages=[
                {"role": "system", "content": "You are an IT support agent helping users with their technical issues."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    def generate_reply(self, message, sender, **kwargs):
        """
        Step 3: Generates a response based on user input and past issues.
        - Retrieves past issues if available
        - Stores the latest issue in memory
        - Calls the GPT model to generate a response
        """

        context = self.memory.get(sender, "")
        self.memory[sender] = message  # Store the latest issue
        return self._get_gpt_response(message, context)



# Step 5: Streamlit UI for real-time chatbot interaction
st.title(" IT Support Chatbot")
st.write("Ask me about your IT issues, and I'll provide troubleshooting steps!")

# Initialize chatbot
if "chatbot" not in st.session_state:
    st.session_state.chatbot = ITSupportBot(name="HelpDeskBot")

# Input field for user query
user_input = st.text_input("You:", "")


if st.button("Send"):
    if user_input:
        response = st.session_state.chatbot.generate_reply(user_input, "User1")
        st.write(f"**HelpDeskBot:** {response}")