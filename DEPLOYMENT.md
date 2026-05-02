# Deployment Guide

The fastest deployment path for this project is Streamlit Community Cloud.

## Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/ and sign in with GitHub.
2. Select **New app**.
3. Choose this repository:
   `gagandeepsingh76/AI-Chatbot-with-Context-Aware-Responses-RAG-based-`
4. Set the branch to `main`.
5. Set the main file path to `app.py`.
6. Open **Advanced settings** and add this secret:

```toml
OPENAI_API_KEY = "sk-your-openai-api-key"
```

7. Click **Deploy**.

After deployment, Streamlit gives you a public URL that normal users can open in a browser.

## API Key Options

- If `OPENAI_API_KEY` is added in Streamlit secrets, visitors can use the chatbot without entering their own key.
- If no hosted key is configured, visitors can still use demo mode or enter their own OpenAI API key in the sidebar.
- Do not commit a real `.streamlit/secrets.toml` file to GitHub.

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
