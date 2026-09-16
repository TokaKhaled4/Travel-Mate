import os
import base64
import mimetypes
import requests
import httpx
import streamlit as st

from typing import Optional
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

st.sidebar.caption(
    "Your tokens are used only for this app session."
)


# ============================================================
# CLEAR CHAT
# ============================================================

if st.sidebar.button("🗑️ Clear Chat"):

    st.session_state.messages = []
    st.session_state.memory = InMemorySaver()
    st.session_state.thread_id = "travel_chat"

    st.rerun()


# ============================================================
# API KEY
# ============================================================

if not openrouter_key:

    st.title("🌍 TravelMate")

    st.info(
        "Enter your OpenRouter API token in the sidebar "
        "to start chatting."
    )

    st.stop()


os.environ["OPENROUTER_API_KEY"] = openrouter_key

RESTCOUNTRIES_API_KEY = restcountries_key


# ============================================================
# MODEL
# Same model as notebook
# ============================================================

model = init_chat_model(
    "google/gemini-2.5-flash-lite",
    model_provider="openrouter",
    temperature=0.2,
)


# ============================================================
# STRUCTURED OUTPUT
# Same schema as notebook
# ============================================================

class TripRequest(BaseModel):

    destination: Optional[str] = Field(
        default=None,
        description=(
            "The country or city the user wants to visit"
        )
    )

    duration_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=60,
        description="Trip duration in days"
    )

    budget_level: Optional[str] = Field(
        default=None,
        description=(
            "Budget category: budget, moderate, or luxury"
        )
    )

    interests: list[str] = Field(
        default_factory=list,
        description=(
            "Travel interests such as history, "
            "beaches, food, nature"
        )
    )

    preferred_weather: Optional[str] = Field(
        default=None,
        description="Preferred weather conditions"
    )


structured_model = model.with_structured_output(
    TripRequest
)


# ============================================================
# WEATHER TOOL
# Same tool as notebook
# ============================================================

@tool
def get_weather(city: str) -> str:
    """
    Get the current weather and today's forecast for a city.
    Use this when the user asks about weather or travel conditions.
    """

    try:

        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
        )

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

            return (
                f"Could not find the location: {city}"
            )

        location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        name = location["name"]
        country = location.get("country", "")

        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
        )

        weather_response = requests.get(
            weather_url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "weather_code,"
                    "wind_speed_10m"
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

        return (
            f"Weather API error: {str(e)}"
        )

    except Exception as e:

        return (
            f"Unexpected weather tool error: {str(e)}"
        )


# ============================================================
# COUNTRY TOOL
# Same tool as notebook
# ============================================================

def _extract_names(
    value,
    key_candidates=("name", "code", "common"),
):

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

                for k in key_candidates:

                    if k in item:

                        parts.append(
                            str(item[k])
                        )

                        break

                else:

                    parts.append(
                        str(item)
                    )

        return ", ".join(parts)

    return str(value)


@tool
def get_country_info(country_name: str) -> str:
    """
    Get basic facts about a country:
    capital, currency, languages, and region.

    Uses the REST Countries v5 API.
    """

    if not RESTCOUNTRIES_API_KEY:

        return (
            "REST Countries API token is missing. "
            "Please enter it in the sidebar."
        )

    try:

        resp = httpx.get(
            (
                "https://api.restcountries.com/"
                f"countries/v5/names.common/{country_name}"
            ),
            headers={
                "Authorization":
                    f"Bearer {RESTCOUNTRIES_API_KEY}"
            },
            timeout=10,
            follow_redirects=True,
        )

        resp.raise_for_status()

        payload = resp.json()

        if "errors" in payload:

            msg = payload["errors"][0].get(
                "message",
                "Unknown error"
            )

            return (
                f"Country API error: {msg}"
            )

        objects = (
            payload
            .get("data", {})
            .get("objects", [])
        )

        if not objects:

            return (
                f"No country found matching "
                f"'{country_name}'."
            )

        data = objects[0]

        capitals = data.get(
            "capitals",
            []
        )

        if (
            capitals
            and isinstance(
                capitals[0],
                dict
            )
        ):

            capital = capitals[0].get(
                "name",
                "Unknown"
            )

        elif capitals:

            capital = capitals[0]

        else:

            capital = "Unknown"

        currencies = _extract_names(
            data.get("currencies"),
            key_candidates=("name", "code")
        )

        languages = _extract_names(
            data.get("languages"),
            key_candidates=("name",)
        )

        region = data.get(
            "region",
            "Unknown"
        )

        return (
            f"Capital: {capital}, "
            f"Currencies: {currencies}, "
            f"Languages: {languages}, "
            f"Region: {region}"
        )

    except httpx.HTTPStatusError as e:

        return (
            f"Country API error: "
            f"{e.response.status_code} — "
            f"{e.response.text[:200]}"
        )

    except httpx.HTTPError as e:

        return (
            f"Country API network error: {e}"
        )

    except (
        KeyError,
        IndexError,
        ValueError
    ) as e:

        return (
            f"Country data malformed: {e}"
        )


# ============================================================
# TOOLS
# ============================================================

tools = [
    get_weather,
    get_country_info
]


# ============================================================
# MEMORY
# Same as notebook
# ============================================================

if "memory" not in st.session_state:

    st.session_state.memory = (
        InMemorySaver()
    )

if "thread_id" not in st.session_state:

    st.session_state.thread_id = (
        "travel_chat"
    )


memory = st.session_state.memory

thread_id = st.session_state.thread_id


# ============================================================
# SYSTEM PROMPT
# Same as notebook
# ============================================================

SYSTEM_PROMPT = """
You are TravelMate, an AI travel planning assistant.

You help users plan trips and answer travel-related questions.

You have access to these tools:

1. get_weather
2. get_country_info

========================
TOOL USAGE RULES
========================

WEATHER:

- If the user asks about weather, temperature,
  climate conditions, forecast, rain, or similar
  information, ALWAYS call get_weather.
- Never answer a weather question from your own knowledge.

COUNTRY INFORMATION:

- If the user asks for country facts such as:
  capital, currency, language, region, country information,
  or similar factual information, ALWAYS call get_country_info.
- NEVER answer these questions from your own knowledge.
- Even if you already know the answer, you MUST call get_country_info.
- If the user says "that country", "that place", "there", etc.,
  resolve the reference using the previous conversation and then
  call get_country_info for the resolved country.

========================
MEMORY RULES
========================

- You have conversation memory.
- ALWAYS use previous conversation messages when answering
  follow-up questions.
- If the user refers to something they previously mentioned,
  retrieve it from the conversation history.
- NEVER ask the user to repeat information that already exists
  in the conversation.
- Current structured information is supplemental context only.
- An empty field in the current structured extraction does NOT mean
  that the user never provided that information.
- Preserve previously provided destination, duration, budget,
  interests, and preferences throughout the conversation.

========================
TRAVEL PLANNING RULES
========================

- Personalize recommendations using information already provided.
- If enough information exists, make a useful recommendation.
- If information is genuinely missing, ask for it.
- Do not invent tool results.
- When a tool is required by the rules above, actually call the tool
  before answering.
"""


# ============================================================
# AGENT
# Same as notebook
# ============================================================

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory
)


# ============================================================
# IMAGE FUNCTIONS
# Same logic as notebook
# ============================================================

def encode_image(uploaded_file):

    return base64.b64encode(
        uploaded_file.getvalue()
    ).decode("utf-8")


def get_image_data_url(uploaded_file):

    mime_type = uploaded_file.type

    if mime_type is None:

        mime_type = (
            mimetypes.guess_type(
                uploaded_file.name
            )[0]
        )

    if mime_type is None:

        mime_type = "image/jpeg"

    if not mime_type.startswith(
        "image/"
    ):

        raise ValueError(
            "The uploaded file does not "
            "appear to be an image."
        )

    image_base64 = encode_image(
        uploaded_file
    )

    return (
        f"data:{mime_type};base64,"
        f"{image_base64}"
    )


# ============================================================
# VALIDATION
# Same as notebook
# ============================================================

def validate_user_input(user_input):

    if not isinstance(
        user_input,
        str
    ):

        raise TypeError(
            "User input must be text."
        )

    user_input = user_input.strip()

    if not user_input:

        raise ValueError(
            "User input cannot be empty."
        )

    if len(user_input) > 5000:

        raise ValueError(
            "User input is too long. "
            "Please keep it under 5000 characters."
        )

    return user_input


# ============================================================
# STRUCTURED OUTPUT CONNECTION
# Same as notebook
# ============================================================

def trip_request_to_context(
    trip_request: TripRequest
) -> str:

    interests = (
        ", ".join(
            trip_request.interests
        )
        if trip_request.interests
        else "Not specified"
    )

    return f"""
Structured travel information extracted from the user's request:

Destination:
{trip_request.destination or "Not specified"}

Duration:
{trip_request.duration_days or "Not specified"} days

Budget:
{trip_request.budget_level or "Not specified"}

Interests:
{interests}

Preferred weather:
{trip_request.preferred_weather or "Not specified"}
"""


# ============================================================
# LCEL
# Same as notebook
# ============================================================

formatter_prompt = PromptTemplate.from_template(
    """
You are an invisible text reformatter.

Reformat the travel assistant response below
to be clear, clean, and organized with bullet
points where appropriate.

CRITICAL RULES:

1. Do NOT provide options or multiple versions.
2. Do NOT add commentary, explanations, or setups
   such as "Here is the formatted version:".
3. Output ONLY the finalized, reformatted travel
   assistant response directly.

Raw Response:
{response}
"""
)


output_formatting_pipeline = (
    {"response": RunnablePassthrough()}
    | formatter_prompt
    | model
    | StrOutputParser()
)


# ============================================================
# RESPONSE NORMALIZATION
# ============================================================

def normalize_response(
    raw_response
):

    if isinstance(
        raw_response,
        str
    ):

        return raw_response

    if isinstance(
        raw_response,
        list
    ):

        text_parts = []

        for block in raw_response:

            if isinstance(
                block,
                dict
            ):

                if block.get(
                    "type"
                ) == "text":

                    text_parts.append(
                        block.get(
                            "text",
                            ""
                        )
                    )

            elif isinstance(
                block,
                str
            ):

                text_parts.append(
                    block
                )

        return "\n".join(
            text_parts
        )

    return str(
        raw_response
    )


# ============================================================
# RUN AGENT
# Directly adapted from notebook
# ============================================================

def run_agent(
    user_input,
    thread_id="travel_chat",
    uploaded_image=None
):

    try:

        # =====================================
        # 1. Validate input
        # =====================================

        user_input = (
            validate_user_input(
                user_input
            )
        )


        # =====================================
        # 2. Memory configuration
        # =====================================

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }


        # =====================================
        # 3. Extract structured information
        # =====================================

        try:

            trip_request = (
                structured_model.invoke(
                    user_input
                )
            )

            structured_context = (
                trip_request_to_context(
                    trip_request
                )
            )

        except Exception as e:

            print(
                "Structured output warning:",
                e
            )

            structured_context = ""


        # =====================================
        # 4. Do not force empty fields
        # into follow-up questions
        # =====================================

        useful_structured_info = []

        if trip_request.destination:

            useful_structured_info.append(
                f"Destination: "
                f"{trip_request.destination}"
            )

        if trip_request.duration_days:

            useful_structured_info.append(
                f"Duration: "
                f"{trip_request.duration_days} days"
            )

        if trip_request.budget_level:

            useful_structured_info.append(
                f"Budget: "
                f"{trip_request.budget_level}"
            )

        if trip_request.interests:

            useful_structured_info.append(
                "Interests: "
                + ", ".join(
                    trip_request.interests
                )
            )

        if trip_request.preferred_weather:

            useful_structured_info.append(
                "Preferred weather: "
                + trip_request.preferred_weather
            )


        # =====================================
        # 5. Build structured context
        # =====================================

        if useful_structured_info:

            structured_context = (
                "Structured information extracted "
                "from the current user message:\n"
                + "\n".join(
                    useful_structured_info
                )
                + "\n\n"
            )

        else:

            structured_context = (
                "No new structured trip information "
                "was provided in this message.\n\n"
            )


        # =====================================
        # 6. Build text content
        # =====================================

        text_content = (
            structured_context
            + "IMPORTANT:\n"
            + "Use the previous conversation history "
            + "to answer follow-up questions. "
            + "Do not ask the user to repeat information "
            + "that they already provided.\n\n"
            + "Current user message:\n"
            + user_input
        )


        # =====================================
        # 7. TEXT ONLY
        # =====================================

        if uploaded_image is None:

            agent_message = {
                "role": "user",
                "content": text_content
            }


        # =====================================
        # 8. TEXT + IMAGE
        # =====================================

        else:

            image_data_url = (
                get_image_data_url(
                    uploaded_image
                )
            )

            agent_message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": text_content
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            )


        # =====================================
        # 9. Run agent
        # =====================================

        result = agent.invoke(
            {
                "messages": [
                    agent_message
                ]
            },
            config
        )


        # =====================================
        # 10. Extract response
        # =====================================

        messages = result.get(
            "messages",
            []
        )

        if not messages:

            return (
                "TravelMate did not return "
                "a response."
            )


        raw_response = (
            messages[-1].content
        )


        # =====================================
        # 11. Normalize response
        # =====================================

        raw_response = normalize_response(
            raw_response
        )


        # =====================================
        # 12. LCEL formatting
        # =====================================

        formatted_response = (
            output_formatting_pipeline.invoke(
                raw_response
            )
        )

        return formatted_response


    except ValueError as e:

        return (
            f"Invalid input: {e}"
        )


    except Exception as e:

        return (
            f"Agent error: "
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# HEADER
# ============================================================

st.title("🌍 TravelMate")

st.caption(
    "Your AI travel planning assistant"
)


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        if message.get(
            "image"
        ) is not None:

            st.image(
                message["image"]
            )

        st.markdown(
            message["content"]
        )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_image = st.file_uploader(
    "📷 Attach an image (optional)",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp"
    ],
    key="travelmate_image"
)


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Where would you like to travel?"
)


# ============================================================
# CHAT PROCESSING
# ============================================================

if user_input:

    try:

        user_input = (
            validate_user_input(
                user_input
            )
        )


        # ========================================
        # Save uploaded image
        # ========================================

        image_bytes = None

        if uploaded_image is not None:

            image_bytes = (
                uploaded_image.getvalue()
            )


        # ========================================
        # Display user message
        # ========================================

        with st.chat_message(
            "user"
        ):

            st.markdown(
                user_input
            )

            if uploaded_image is not None:

                st.image(
                    uploaded_image
                )


        # ========================================
        # Save user message
        # ========================================

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input,
                "image": image_bytes
            }
        )


        # ========================================
        # Run TravelMate
        # ========================================

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "TravelMate is thinking..."
            ):

                response = run_agent(
                    user_input=user_input,
                    thread_id=thread_id,
                    uploaded_image=uploaded_image
                )


            st.markdown(
                response
            )


        # ========================================
        # Save assistant response
        # ========================================

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )


        # ========================================
        # Clear image uploader
        # ========================================

        st.session_state.pop(
            "travelmate_image",
            None
        )

        st.rerun()


    except ValueError as e:

        st.error(
            f"Invalid input: {e}"
        )


    except Exception as e:

        st.error(
            f"TravelMate error: "
            f"{type(e).__name__}: {e}"
        )
