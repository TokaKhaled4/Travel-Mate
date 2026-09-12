import os
import base64
import requests
import httpx
import streamlit as st

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TravelMate",
    page_icon="✈️",
    layout="centered",
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ TravelMate Settings")

openrouter_key = st.sidebar.text_input(
    "OpenRouter API Token",
    type="password",
    placeholder="Enter your OpenRouter token",
)

restcountries_key = st.sidebar.text_input(
    "REST Countries API Token",
    type="password",
    placeholder="Enter your REST Countries token",
)

st.sidebar.caption("Your tokens are used only for this app session.")

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.session_state.memory = InMemorySaver()
    st.session_state.thread_id = f"travel_chat_{id(st.session_state)}"
    st.rerun()


# ============================================================
# STOP UNTIL OPENROUTER TOKEN IS PROVIDED
# ============================================================

if not openrouter_key:
    st.title("✈️ TravelMate")
    st.info("Enter your OpenRouter API token in the sidebar to start chatting.")
    st.stop()


# ============================================================
# API CONFIGURATION
# ============================================================

os.environ["OPENROUTER_API_KEY"] = openrouter_key
RESTCOUNTRIES_API_KEY = restcountries_key


# ============================================================
# MODEL
# ============================================================

model = init_chat_model(
    "google/gemini-2.5-flash-lite",
    model_provider="openrouter",
    temperature=0.2,
)


# ============================================================
# TOOLS
# ============================================================

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather and today's forecast for a city.
    Use this when the user asks about weather or travel conditions.
    """
    try:
        # Geocoding
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_response = requests.get(
            geo_url,
            params={
                "name": city,
                "count": 1,
                "language": "en",
                "format": "json",
            },
            timeout=10,
        )

        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if not geo_data.get("results"):
            return f"Could not find the location: {city}"

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]
        name = location["name"]
        country = location.get("country", "")

        # Weather
        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_response = requests.get(
            weather_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,relative_humidity_2m,"
                    "weather_code,wind_speed_10m"
                ),
                "timezone": "auto",
            },
            timeout=10,
        )

        weather_response.raise_for_status()
        weather = weather_response.json()["current"]

        return (
            f"Weather for {name}, {country}: "
            f"temperature={weather['temperature_2m']}°C, "
            f"humidity={weather['relative_humidity_2m']}%, "
            f"wind speed={weather['wind_speed_10m']} km/h, "
            f"weather code={weather['weather_code']}."
        )

    except requests.RequestException as e:
        return f"Weather API error: {str(e)}"

    except Exception as e:
        return f"Unexpected weather tool error: {str(e)}"


def _extract_names(
    value,
    key_candidates=("name", "code", "common"),
):
    """Normalize a field that could be a dict, list of dicts,
    or list of strings."""
    if value is None:
        return ""

    if isinstance(value, dict):
        return ", ".join(value.keys())

    if isinstance(value, list):
        parts = []

        for item in value:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                for key in key_candidates:
                    if key in item:
                        parts.append(str(item[key]))
                        break
                else:
                    parts.append(str(item))

        return ", ".join(parts)

    return str(value)


@tool
def get_country_info(country_name: str) -> str:
    """
    Get basic facts about a country: capital, currency,
    languages, and region.
    Uses the REST Countries v5 API.
    """
    if not RESTCOUNTRIES_API_KEY:
        return (
            "REST Countries API token is missing. "
            "Please enter it in the sidebar."
        )

    try:
        resp = httpx.get(
            f"https://api.restcountries.com/countries/v5/names.common/{country_name}",
            headers={
                "Authorization": f"Bearer {RESTCOUNTRIES_API_KEY}"
            },
            timeout=10,
            follow_redirects=True,
        )

        resp.raise_for_status()
        payload = resp.json()

        if "errors" in payload:
            msg = payload["errors"][0].get(
                "message",
                "Unknown error",
            )
            return f"Country API error: {msg}"

        objects = payload.get("data", {}).get("objects", [])

        if not objects:
            return f"No country found matching '{country_name}'."

        data = objects[0]

        capitals = data.get("capitals", [])

        if capitals and isinstance(capitals[0], dict):
            capital = capitals[0].get("name", "Unknown")
        elif capitals:
            capital = capitals[0]
        else:
            capital = "Unknown"

        currencies = _extract_names(
            data.get("currencies"),
            key_candidates=("name", "code"),
        )

        languages = _extract_names(
            data.get("languages"),
            key_candidates=("name",),
        )

        region = data.get("region", "Unknown")

        return (
            f"Capital: {capital}, "
            f"Currencies: {currencies}, "
            f"Languages: {languages}, "
            f"Region: {region}"
        )

    except httpx.HTTPStatusError as e:
        return (
            f"Country API error: {e.response.status_code} — "
            f"{e.response.text[:200]}"
        )

    except httpx.HTTPError as e:
        return f"Country API network error: {e}"

    except (KeyError, IndexError, ValueError) as e:
        return f"Country data malformed: {e}"


tools = [
    get_weather,
    get_country_info,
]


# ============================================================
# MEMORY
# ============================================================

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "travel_chat"

memory = st.session_state.memory
thread_id = st.session_state.thread_id


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are TravelMate, an AI travel planning assistant.

Your job is to help users plan trips.

You have access to:
1. get_weather
2. get_country_info

Use tools when they provide information that you cannot reliably know
from your own knowledge.

Rules:
- Use get_weather for weather-related questions.
- Use get_country_info for country facts.
- Do not call tools unnecessarily.
- Remember relevant information from the conversation.
- Personalize recommendations using the user's previous preferences.
- Be honest when information is unavailable.
- Never invent API results.
"""


# ============================================================
# AGENT
# ============================================================

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory,
)


# ============================================================
# VALIDATION
# ============================================================

def validate_user_input(user_input):
    if not isinstance(user_input, str):
        raise TypeError("User input must be text.")

    user_input = user_input.strip()

    if not user_input:
        raise ValueError("User input cannot be empty.")

    if len(user_input) > 5000:
        raise ValueError(
            "User input is too long. Please keep it under 5000 characters."
        )

    return user_input


# ============================================================
# IMAGE
# ============================================================

def image_to_data_url(uploaded_file):
    image_bytes = uploaded_file.getvalue()
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    return f"data:{uploaded_file.type};base64,{encoded}"


# ============================================================
# SESSION CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HEADER
# ============================================================

st.title("🌍 TravelMate")
st.caption("Your AI travel planning assistant")


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message.get("image") is not None:
            st.image(message["image"])

        st.markdown(message["content"])


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_image = st.file_uploader(
    "📷 Attach an image (optional)",
    type=["png", "jpg", "jpeg", "webp"],
    key="image_uploader",
)


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Where would you like to travel?"
)


# ============================================================
# PROCESS MESSAGE
# ============================================================

if user_input:

    try:
        user_input = validate_user_input(user_input)

        # ----------------------------------------------------
        # Display user message
        # ----------------------------------------------------

        with st.chat_message("user"):

            st.markdown(user_input)

            if uploaded_image is not None:
                st.image(uploaded_image)

        # ----------------------------------------------------
        # Save UI history
        # ----------------------------------------------------

        image_bytes = (
            uploaded_image.getvalue()
            if uploaded_image is not None
            else None
        )

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input,
                "image": image_bytes,
            }
        )

        # ----------------------------------------------------
        # Build agent message
        # ----------------------------------------------------

        if uploaded_image is not None:

            image_data_url = image_to_data_url(uploaded_image)

            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": user_input,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    },
                ]
            )

        else:

            message = {
                "role": "user",
                "content": user_input,
            }

        # ----------------------------------------------------
        # Agent config
        # ----------------------------------------------------

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # ----------------------------------------------------
        # Run agent
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner("TravelMate is thinking..."):

                result = agent.invoke(
                    {
                        "messages": [message]
                    },
                    config,
                )

            response = result["messages"][-1].content

            # Handle models that return structured content
            if not isinstance(response, str):
                response = str(response)

            st.markdown(response)

        # ----------------------------------------------------
        # Save assistant response
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        # Clear uploaded image after sending
        st.session_state.pop("image_uploader", None)

    except ValueError as e:

        st.error(f"Invalid input: {e}")

    except Exception as e:

        st.error(f"TravelMate error: {e}")
