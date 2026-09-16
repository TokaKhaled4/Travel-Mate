# 🌍 TravelMate

TravelMate is an AI-powered travel planning assistant that helps users plan personalized trips using conversational context, structured travel preferences, external travel APIs, and multimodal image input.

It can remember information such as a user's destination, trip duration, budget, interests, and preferred weather across conversation turns, then use that context to provide more personalized travel assistance.

## 🚀 Live Demo

TravelMate is deployed using **Streamlit** and is available here:

**[Open TravelMate](https://travel-mate-pbdsvcli7snxvzh7sakj2l.streamlit.app/)**

---

## 🤖 Model & Provider

- **Model:** `google/gemini-2.5-flash-lite`
- **Provider:** OpenRouter
- **API authentication:** `OPENROUTER_API_KEY`

---

## 🛠️ APIs & Tools

TravelMate currently exposes two tools to the agent:

### 1. Weather — `get_weather`

Uses the **Open-Meteo Geocoding API** to resolve a city and the **Open-Meteo Forecast API** to retrieve current weather information.

The tool returns:

- Temperature
- Relative humidity
- Wind speed
- Weather code
- Resolved city and country

The agent is instructed to call this tool whenever the user asks about weather, temperature, climate conditions, forecasts, rain, or similar information.

### 2. Country Information — `get_country_info`

Uses the **REST Countries API** to retrieve country facts.

The tool provides:

- Capital
- Currency
- Languages
- Region

The agent is instructed to call this tool for country-information questions rather than answering those facts directly from model knowledge.

---

## 🧠 Memory

TravelMate uses **LangGraph's `InMemorySaver`** as its checkpointer:

```python
memory = InMemorySaver()
```

The agent is created with this checkpointer:

```python
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory
)
```

Conversation memory is associated with a `thread_id`:

```python
config = {
    "configurable": {
        "thread_id": thread_id
    }
}
```

This allows messages from previous turns in the same conversation thread to be available to the agent.

### What memory is used for

The agent is instructed to preserve previously provided:

- Destination
- Trip duration
- Budget
- Interests
- Preferences

For example, if a user first says:

> "I'm planning a trip to Greece for 7 days. My budget is moderate and I love history, museums, and food."

They can later ask:

> "What interests did I tell you I have?"

TravelMate can answer using the conversation history without asking the user to repeat the information.

### Important

The current implementation uses **in-memory conversation storage**, so memory is not a permanent database. It is intended for the active application/session rather than durable long-term user storage.

---

## 🔄 Agent Workflow & Tool-Calling Flow

The main processing flow is:

```text
User message
     │
     ▼
Input validation
     │
     ▼
Structured extraction
     │
     ▼
TripRequest → context text
     │
     ▼
Conversation history + current context
     │
     ▼
Multimodal message construction
     │
     ▼
TravelMate LangGraph agent
     │
     ├──► get_weather ───────► Open-Meteo
     │
     ├──► get_country_info ─► REST Countries
     │
     └──► Direct response when no external tool is required
     │
     ▼
Raw agent response
     │
     ▼
LCEL formatting pipeline
     │
     ▼
Final response
```

### Step-by-step

1. **Validate the user input**
   - Input must be text.
   - Empty messages are rejected.
   - Messages longer than 5,000 characters are rejected.

2. **Extract structured travel information**
   - The current user message is passed to a structured-output version of the model.
   - The result is represented as a `TripRequest` Pydantic object.

3. **Connect structured output to the agent**
   - Only fields actually provided in the current message are added to the structured context.
   - Empty fields do not overwrite information already available in conversation memory.

4. **Build the agent message**
   - The structured context is combined with instructions to use previous conversation history.
   - The current user message is then included.

5. **Handle optional image input**
   - Text-only requests use a normal text message.
   - Image requests use a multimodal message containing text and an image URL.

6. **Run the agent**
   - The LangGraph agent receives the message and the thread configuration.
   - The model decides whether one of the available tools is required.

7. **Tool calling**
   - Weather-related requests trigger `get_weather`.
   - Country-fact requests trigger `get_country_info`.
   - Tool results are returned to the agent as tool messages.

8. **Generate the final answer**
   - The agent uses the tool results and conversation history to produce the response.

9. **Post-process the response**
   - The raw response is passed through an LCEL formatting pipeline to make the final answer clearer and more organized.

---

## 🧩 Structured Output Schema

TravelMate uses a Pydantic schema named `TripRequest`:

```python
class TripRequest(BaseModel):
    destination: Optional[str] = Field(
        default=None,
        description="The country or city the user wants to visit"
    )

    duration_days: Optional[int] = Field(
        default=None,
        ge=1,
        le=60,
        description="Trip duration in days"
    )

    budget_level: Optional[str] = Field(
        default=None,
        description="Budget category: budget, moderate, or luxury"
    )

    interests: list[str] = Field(
        default_factory=list,
        description="Travel interests such as history, beaches, food, nature"
    )

    preferred_weather: Optional[str] = Field(
        default=None,
        description="Preferred weather conditions"
    )
```

The schema is connected to the model with:

```python
structured_model = model.with_structured_output(TripRequest)
```

The extracted object is then converted into contextual text using `trip_request_to_context()` and passed into the agent workflow.

### Example

A request such as:

```text
I want to visit Italy for 7 days.
I have a moderate budget.
I love history, museums and Italian food.
I prefer mild weather.
```

can be represented as:

```text
destination = "Italy"
duration_days = 7
budget_level = "moderate"
interests = ["history", "museums", "food"]
preferred_weather = "mild"
```

---

## 🔗 LCEL / Runnables

TravelMate uses LangChain Expression Language (**LCEL**) for the response post-processing pipeline.

The notebook imports:

```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
```

The implemented formatting pipeline is:

```python
output_formatting_pipeline = (
    {"response": RunnablePassthrough()}
    | formatter_prompt
    | model
    | StrOutputParser()
)
```

### Pipeline flow

```text
Raw agent response
       │
       ▼
RunnablePassthrough
       │
       ▼
PromptTemplate
       │
       ▼
Gemini model
       │
       ▼
StrOutputParser
       │
       ▼
Formatted response
```

The LCEL pipeline is used specifically to reformat the generated travel response into a clearer and more organized final response.

---

## 🖼️ Multimodal Input

TravelMate supports **text + image** input.

Images are:

1. Read from the uploaded file.
2. Encoded using Base64.
3. Converted into a data URL.
4. Added to the LangChain message as an `image_url` content block.

The message structure is conceptually:

```python
{
    "role": "user",
    "content": [
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
}
```

This allows the multimodal model to analyze an uploaded travel image while also considering the user's text and previous conversation context.

For example, a user can upload a picture of a landmark and ask:

> "What is this place, and would I enjoy visiting it based on what you know about me?"

The model can combine visual information with the user's remembered travel interests.


---

## ⚙️ Setup & Run

### 1. Clone the project

```bash
git clone <your-repository-url>
cd TravelMate
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

The notebook uses:

```bash
pip install -U langchain langchain-openrouter langgraph pydantic requests httpx streamlit
```

### 4. Configure API keys

Set your OpenRouter API key.

**Windows PowerShell:**

```powershell
$env:OPENROUTER_API_KEY="your_openrouter_api_key"
```

**macOS/Linux:**

```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

If the Streamlit version uses Streamlit secrets, add the key to:

```text
.streamlit/secrets.toml
```

with:

```toml
OPENROUTER_API_KEY = "your_openrouter_api_key"
```

The country-information tool also expects a `RESTCOUNTRIES_API_KEY` value in the notebook implementation.

### 5. Run the Streamlit app

If the application entry point is `app.py`:

```bash
streamlit run app.py
```

The application will then be available through the local Streamlit URL shown in the terminal.

---

## ⚠️ Known Limitations

- **In-memory memory:** `InMemorySaver` does not provide persistent database-backed memory.
- **Session/thread dependence:** Conversation memory depends on maintaining the same `thread_id`.
- **Limited tools:** The agent currently has weather and country-information tools only.
- **No booking integration:** Flights, hotels, restaurants, and activities are not directly booked by the agent.
- **REST Countries dependency:** Country information depends on the configured REST Countries API and its authentication/endpoint availability.
- **Weather dependency:** Current weather requires the Open-Meteo services to be reachable.
- **Image handling:** Images are Base64-encoded and sent as multimodal input, which can increase request size.
- **Structured extraction is current-message based:** The structured extractor focuses on the current user message; conversation history remains the primary source for preserving previously provided information.
- **Response formatting adds an additional model call:** The LCEL formatting stage sends the generated response through the model again.
- **Input limit:** User text is limited to 5,000 characters by validation.

---

## 🔮 Future Improvements

Potential future improvements include:

- Replace `InMemorySaver` with persistent storage such as PostgreSQL, Redis, or another database.
- Add flight, hotel, attraction, restaurant, and transportation APIs.
- Add itinerary generation with day-by-day scheduling.
- Add budget estimation and currency conversion.
- Add map and route integration.
- Add richer image understanding for landmarks, menus, hotels, and destinations.
- Add user authentication and persistent profiles.
- Add caching for frequently requested API data.
- Add asynchronous API calls for improved response time.
- Add automated evaluation and monitoring of tool calls and generated itineraries.
- Reduce unnecessary model calls by integrating response formatting more efficiently.
- Add stronger validation for `budget_level` using an enum such as `budget`, `moderate`, and `luxury`.

