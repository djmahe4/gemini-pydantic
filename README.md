# Pydantic & GenAI: A Use-Case Explorer

This Streamlit application is an interactive tool designed to demonstrate the powerful combination of Google's Gemini Pro generative AI and Pydantic for data validation. It serves as a hands-on playground to explore how you can transform unstructured text or simple prompts into clean, structured, and validated JSON output for various real-world tasks.

<!-- [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gemini-pydantic.streamlit.app/) -->

## ✨ Key Features

*   **Multi-Use-Case Playground:** Explore several pre-built tasks to see different applications of GenAI.
*   **Dynamic Prompt Engineering:** See how prompts are tailored for specific tasks.
*   **Pydantic Data Validation:** Every output from the AI is rigorously validated against a Pydantic model, ensuring the data is structured, typed, and correct.
*   **Schema-Driven Generation:** The application sends the Pydantic model's JSON schema directly within the prompt, instructing the Gemini model to return a perfectly formatted response.
*   **Secure API Key Management:** Uses a `.env` file to safely manage your Google API key.
*   **Extensible by Design:** Easily add your own custom use cases by defining a new Pydantic model and a prompt template.

---

## 🚀 Use Cases Explored

This application comes with several pre-configured examples:

1.  **Text Summarization:** Condenses a long piece of text into a concise summary and extracts relevant keywords.
2.  **Sentiment Analysis:** Analyzes a block of text to determine if the sentiment is positive, negative, or neutral, and provides a confidence score.
3.  **Entity Extraction (NER):** Identifies and extracts named entities like people, organizations, and locations from text.
4.  **Recipe Generator:** Creates a complete recipe with a dish name, ingredient list, and step-by-step instructions from a simple prompt.
5.  **LLM Training Dataset Generation:** Converts a source text into a list of high-quality Question/Answer pairs, suitable for fine-tuning a language model.

 <!-- Replace with a real screenshot URL -->

---

## 🛠️ Tech Stack

*   **Language:** Python 3.9+
*   **Framework:** [Streamlit](https://streamlit.io/) - for the interactive web UI.
*   **Generative AI:** [Google Gemini Pro](https://deepmind.google/technologies/gemini/) via the `google-generativeai` library.
*   **Data Validation:** [Pydantic V2](https://docs.pydantic.dev/) - for defining data schemas and validating API outputs.
*   **Configuration:** `python-dotenv` - for managing environment variables.

---

## ⚙️ Setup and Installation

Follow these steps to run the application locally.

### 1. Prerequisites
*   Python 3.9 or higher.
*   A Google API Key for the Gemini model. You can obtain one from [Google AI Studio](https://makersuite.google.com/app/apikey).

### 2. Clone the Repository
```bash
git clone https://github.com/djmahe4/gemini-pydantic
cd gemini-pydantic
```

### 3. Set Up Your Environment Variable
Create a file named `.env` in the root of the project directory. Add your Google API key to this file:
```
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```
**Note:** The `.env` file is included in `.gitignore` to prevent you from accidentally committing your secret key.

### 4. Install Dependencies
Install all the required Python libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Once the setup is complete, you can run the Streamlit application with a single command:

```bash
streamlit run streamlit_app.py
```

This will start a local server and open the application in your default web browser.

---

## 💡 How It Works

The application follows a simple but powerful workflow for each use case:

1.  **Select a Use Case:** The user chooses a task from the sidebar (e.g., "Sentiment Analysis").
2.  **Load Configuration:** The app loads the corresponding Pydantic model (`SentimentAnalysis`), a prompt template, and UI labels from a central `USE_CASES` dictionary.
3.  **Inject Schema into Prompt:** The Pydantic model's JSON schema is generated using `.schema_json()`. This schema is embedded directly into the prompt sent to Gemini, instructing the AI on the exact format for its response.
4.  **API Call:** The formatted prompt, including the user's input text, is sent to the Gemini API.
5.  **Validate and Parse:** When Gemini returns a response, the application attempts to parse the JSON string directly into an instance of the `SentimentAnalysis` Pydantic model.
    *   **On Success:** If the response matches the schema, the data is successfully validated and a clean Python object is created.
    *   **On Failure:** If the JSON is malformed or misses required fields, Pydantic raises a validation error, which is caught and displayed to the user.
6.  **Display Results:** The validated, structured data object is displayed in the UI.

---

## 🧩 How to Add a New Use Case

The application is designed to be easily extensible. To add your own custom functionality, follow these two steps:

### Step 1: Define a New Pydantic Model
In `app.py`, define a new class that inherits from `BaseModel`. This class represents the data structure you want the AI to return.

**Example: A "Code Review" model**
```python
class CodeReview(BaseModel):
    is_clean: bool = Field(description="Whether the code follows best practices.")
    suggestions: List[str] = Field(description="A list of specific suggestions for improvement.")
    overall_rating: int = Field(description="A rating from 1 (poor) to 5 (excellent).")
```

### Step 2: Add the Configuration to `USE_CASES`
In `app.py`, add a new key-value pair to the `USE_CASES` dictionary. This entry links your new model to a prompt template and UI text.

```python
# Inside the USE_CASES dictionary
"Code Review": {
    "model": CodeReview,  # Link to your new Pydantic model
    "prompt_template": """
        Act as an expert code reviewer. Analyze the following code snippet.
        Your output must be a JSON object that strictly adheres to this schema:
        ```json
        {schema}
        ```
        Text to analyze:
        ---
        {user_input}
        ---
    """,
    "input_label": "Enter a code snippet for review...",
    "default_input": "def my_func(a,b):\n  return a+ b"
}
```

Save the file, and your Streamlit app will automatically reload with "Code Review" as a new, fully functional option in the dropdown menu.

---

## 📄 License

> _This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details._
