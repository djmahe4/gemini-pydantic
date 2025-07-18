import streamlit as st
import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel, Field, conlist
from typing import List, Type


# --- 1. Pydantic Models for Different Use Cases ---
# We define a structure for each task we want to perform.

class Summary(BaseModel):
    """Pydantic model for a summarized text."""
    summary: str = Field(description="A concise summary of the provided text.")
    keywords: List[str] = Field(description="A list of the most important keywords.")


class SentimentAnalysis(BaseModel):
    """Pydantic model for analyzing sentiment."""
    sentiment: str = Field(description="The overall sentiment (e.g., Positive, Negative, Neutral).")
    confidence_score: float = Field(description="A score from 0.0 to 1.0 indicating confidence.")


class EntityExtraction(BaseModel):
    """Pydantic model for extracting named entities."""
    people: List[str] = Field(description="List of names of people mentioned.")
    organizations: List[str] = Field(description="List of organizations mentioned.")
    locations: List[str] = Field(description="List of locations mentioned.")


class Ingredient(BaseModel):
    """A single ingredient for a recipe."""
    name: str
    quantity: str


class RecipeGenerator(BaseModel):
    """Pydantic model for generating a recipe."""
    dish_name: str = Field(description="The name of the dish.")
    ingredients: List[Ingredient] = Field(description="A list of ingredients with names and quantities.")
    instructions: conlist(str, min_length=3) = Field(description="Step-by-step cooking instructions.")

# --- ADD THIS SNIPPET FOR LLM TRAINING DATA ---

class QAPair(BaseModel):
    """A single question-answer pair."""
    question: str = Field(description="A specific question that can be answered by the text.")
    answer: str = Field(description="The corresponding answer to the question, extracted from the text.")

class LLMTrainingData(BaseModel):
    """A structured dataset for training a Question-Answering LLM."""
    dataset_name: str = Field(description="A descriptive name for the dataset, derived from the text's topic.")
    qa_pairs: conlist(QAPair, min_length=3) = Field(description="A list of question-answer pairs.")

# --- 2. Use Case Configuration ---
# We map a user-friendly name to the corresponding Pydantic model and a prompt template.

USE_CASES = {
    "Text Summarization": {
        "model": Summary,
        "prompt_template": """
            Your task is to act as a text summarization engine.
            Analyze the following text and generate a concise summary and a list of keywords.
            Your output must be a JSON object that strictly adheres to this schema:
            ```json
            {schema}
            ```
            Do not include any text or formatting outside the JSON object.

            Text to analyze:
            ---
            {user_input}
            ---
        """,
        "input_label": "Enter text to summarize...",
        "default_input": "The Apollo 11 mission, launched on July 16, 1969, was the first to land humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969. Armstrong became the first person to step onto the lunar surface six hours and 39 minutes later."
    },
    "Sentiment Analysis": {
        "model": SentimentAnalysis,
        "prompt_template": """
            You are a sentiment analysis expert.
            Analyze the sentiment of the following text and provide a confidence score.
            Your output must be a JSON object that strictly adheres to this schema:
            ```json
            {schema}
            ```
            Do not include any text or formatting outside the JSON object.

            Text to analyze:
            ---
            {user_input}
            ---
        """,
        "input_label": "Enter text for sentiment analysis...",
        "default_input": "I am absolutely thrilled with the new software update! It's fast, intuitive, and has all the features I was hoping for. A huge improvement over the last version."
    },
    "Entity Extraction": {
        "model": EntityExtraction,
        "prompt_template": """
            You are a Named Entity Recognition (NER) system.
            Extract all names of people, organizations, and locations from the following text.
            Your output must be a JSON object that strictly adheres to this schema:
            ```json
            {schema}
            ```
            Do not include any text or formatting outside the JSON object.

            Text to analyze:
            ---
            {user_input}
            ---
        """,
        "input_label": "Enter text to extract entities from...",
        "default_input": "Sundar Pichai, the CEO of Google, announced at a conference in New York that Alphabet Inc. plans to expand its operations in Europe."
    },
    "Recipe Generator": {
        "model": RecipeGenerator,
        "prompt_template": """
            You are a creative chef.
            Generate a simple recipe based on the following main ingredients or dish idea.
            Your output must be a JSON object that strictly adheres to this schema:
            ```json
            {schema}
            ```
            Do not include any text or formatting outside the JSON object.

            Recipe Idea:
            ---
            {user_input}
            ---
        """,
        "input_label": "Enter a dish name or main ingredients (e.g., 'chicken and lemon')...",
        "default_input": "A simple pasta with tomatoes and basil."
    },
    "LLM Training Dataset Generation": {
        "model": LLMTrainingData,
        "prompt_template": """
            You are an expert in creating training data for Language Models.
            Your task is to read the following text and generate a high-quality dataset of question-and-answer pairs based on its content.
            The goal is to create data that could be used to fine-tune a question-answering AI.
            Your output must be a JSON object that strictly adheres to this schema:
            ```json
            {schema}
            ```
            Do not include any text or formatting outside the JSON object.

            Source Text:
            ---
            {user_input}
            ---
        """,
        "input_label": "Enter source text to generate a Q&A dataset from...",
        "default_input": "The Solar System is the gravitationally bound system of the Sun and the objects that orbit it. Of the bodies that orbit the Sun directly, the largest are the eight planets, with the remainder being smaller objects, such as the five dwarf planets and small Solar System bodies. The Solar System formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud."
    }
}

# --- 3. Streamlit Application UI ---
st.set_page_config(page_title="Pydantic & GenAI Explorer", layout="wide")
st.title("🔬 Pydantic & GenAI: A Use-Case Explorer")
st.markdown(
    "Select a use case, provide an input, and see how Gemini can generate structured, validated data using Pydantic.")

# --- API Key Configuration ---
try:
    load_dotenv()
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-2.5-pro')
except (KeyError, TypeError):
    st.error("⚠️ Your Google API Key is not configured. Please create a `.env` file with `GOOGLE_API_KEY='Your_Key'`.")
    st.stop()

# --- Sidebar for selecting the use case ---
with st.sidebar:
    st.header("🛠️ Choose Your Tool")
    selected_use_case = st.selectbox(
        "Select a use case to explore:",
        options=list(USE_CASES.keys())
    )
    st.markdown("---")

    # Get the configuration for the selected use case
    config = USE_CASES[selected_use_case]
    PydanticModel = config["model"]

    with st.expander("Expected Output Structure (Pydantic Schema)", expanded=False):
        st.json(PydanticModel.schema_json(indent=2))

# --- Main Area for Input and Output ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Input")
    user_input = st.text_area(
        label=config["input_label"],
        value=config["default_input"],
        height=250
    )
    if st.button(f"🚀 Generate for {selected_use_case}", use_container_width=True):
        if not user_input:
            st.warning("Please provide an input text.")
        else:
            with st.spinner("Calling Gemini..."):
                try:
                    # Format the prompt with the schema and user input
                    schema_json = PydanticModel.schema_json(indent=2)
                    prompt = config["prompt_template"].format(schema=schema_json, user_input=user_input)

                    # Call the Generative AI model
                    response = model.generate_content(prompt)
                    raw_response_text = response.text

                    # Clean the response to extract the pure JSON part
                    json_str = raw_response_text.strip().lstrip("```json\n").rstrip("\n```")
                    # parts = raw_response_text.split("```json\n", 1)  # Split only once
                    # if len(parts) > 1:
                    #     json_str = parts[1].split("\n```", 1)[0].strip()
                    # else:
                    #     json_str = ""  # Not found

                    # Parse the JSON string into the Pydantic model
                    # This step validates the structure and types
                    parsed_output = PydanticModel.parse_raw(json_str)

                    # Store the result in session state to display in the other column
                    st.session_state.last_result = parsed_output

                except json.JSONDecodeError as e:
                    st.session_state.last_result = {"error": "JSON Decode Error", "details": str(e),
                                                    "raw_response": raw_response_text}
                except Exception as e:
                    st.session_state.last_result = {"error": "Validation or API Error", "details": str(e),
                                                    "raw_response": raw_response_text}

with col2:
    st.subheader("✅ Structured Output (Validated by Pydantic)")
    if "last_result" in st.session_state:
        result = st.session_state.last_result
        if isinstance(result, BaseModel):
            st.success("Successfully generated and validated!")
            st.write(result)
        elif "error" in result:
            st.error(f"**{result['error']}**")
            st.code(result['details'], language='text')
            with st.expander("Model's Raw Response"):
                st.text(result['raw_response'])
    else:
        st.info("Click the 'Generate' button to see the output here.")