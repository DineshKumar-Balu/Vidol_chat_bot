
import streamlit as st
import json
from google.oauth2.service_account import Credentials
from vertexai.language_models import TextGenerationModel
import vertexai

# Function to load language model
def load_language_model():
    key_path = 'vertixai-poc-413616-4b5a6e92b31d.json'
    credentials = Credentials.from_service_account_file(key_path)
    PROJECT_ID = 'vertixai-poc-413616'
    REGION = 'us-central1'
    vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)
    model = TextGenerationModel.from_pretrained("text-bison-32k")
    return model

# Function to make predictions
def generate_response(model, input_text):
    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": 0.9,
        "top_p": 1,
    }
    response = model.predict(input_text, **parameters)
    return response.text

# Function to get conversational chain
def get_conversational_chain(context, question):
    prompt = f"Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, 'I appreciate your inquiry. Unfortunately, the information needed to address your question is not available in the provided context. If you could provide additional details or specify any particular points you'd like to discuss, I would be more than happy to assist you to the best of my abilities.', don't provide the wrong answer\n\nContext:\n {context}?\nQuestion:\n{question}\n\nAnswer: "
    return prompt

# Main Streamlit app
def main():
    st.set_page_config("ChimeraAI",page_icon='chimeraAI/chimera-logo.jpg')
    # st.write("<p style='font-size: 30px'>chimeraAI</p>",unsafe_allow_html=True)
    st.write("---")
    st.write("<p style='font-size: 50px'>www.chimeratechnologies.com</p>",unsafe_allow_html=True)
    # st.write("###")
    # st.title("Language Model Chatbot")

    # Load language model
    model = load_language_model()
    

    # Specify the path to your JSON file
    json_file_path = "tt.json"

    # Read the JSON file
    with open(json_file_path, "r") as file:
        data = json.load(file)

    # User input
    user_input = st.text_input("Ask me a question:")

    if user_input:
        # Include the input text in the format for predictions
        input_text = f"{data}\n{user_input}\n"

        # Generate response
        response = generate_response(model, get_conversational_chain(data, user_input))

        # Display response
        st.text("ChimeraAI:")
        st.write(response)
        st.write("---")

if __name__ == "__main__":
    main()
