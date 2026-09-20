import os

from dotenv import load_dotenv
from loguru import logger

# LangChain and Deep Agents imports
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from rags import retrieve

# Pipecat frame and processor imports
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
)
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
)

# Load OPENAI_API_KEY and other configuration from .env
load_dotenv()

# ============================================================================
# 1. TOOL DEFINITIONS
# ============================================================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Replace this mock result with a real weather API in production.
    return f"The weather in {city} is 25 degrees Celsius and very cool."


# ============================================================================
# 1b. CONTENT NORMALIZATION
# ============================================================================

def extract_text(content) -> str:
    """Return the spoken-text portion of a LangChain message content value.

    With ``use_responses_api=True`` the OpenAI Responses API does not stream
    plain strings. Each chunk's ``content`` is a list of typed blocks, e.g.::

        [{"type": "reasoning", "summary": [], ...}]
        [{"type": "text", "text": "Hello", "phase": "final_answer", ...}]

    Only ``text`` blocks carry words meant for the user; ``reasoning`` blocks
    are internal and must never be sent to TTS. Checking
    ``isinstance(content, str)`` silently discards everything under the
    Responses API, which is why the bot stayed silent.

    Handles the Chat Completions shape (a bare string) as well, so the
    processor keeps working if ``use_responses_api`` is turned off.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    return ""


# ============================================================================
# 1c. AGENT FACTORY
# ============================================================================
# Building the agent in one function keeps the model, the tools and the prompt
# in a single place, so the pipeline processor below stays about frame handling.

SYSTEM_PROMPT = (
    "You are a helpful bilingual English and Arabic voice assistant. "
    "Always reply in the same language as the user's latest message. "
    "If the user mixes English and Arabic, reply naturally using the "
    "same language mixture. "
    "Use clear Modern Standard Arabic unless the user uses a "
    "recognizable dialect, in which case you may mirror that dialect. "
    "Keep responses concise, conversational, and suitable for spoken "
    "voice output. "
    "Do not use Markdown formatting, tables, or emoji because your "
    "response will be spoken aloud. "
    "Use the retrieve tool for knowledge-base questions."
)

# Uses gpt-5.5 by default.
# You can override it in .env:
#
# OPENAI_MODEL=gpt-5.5
#
DEFAULT_MODEL = "gpt-5.5"


def model_name() -> str:
    """Resolve the chat model, preferring OPENAI_MODEL from .env."""
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def build_deep_agent(*, checkpointer=None, system_prompt: str = SYSTEM_PROMPT):
    """Create the bilingual Deep Agent used by both the voice and text paths."""
    model = init_chat_model(
        f"openai:{model_name()}",
        use_responses_api=True,
    )

    return create_deep_agent(
        model=model,
        tools=[
            get_weather,
            retrieve,
        ],
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )


# ============================================================================
# 2. PIPECAT PROCESSOR FOR THE DEEP AGENT
# ============================================================================

class DeepAgentProcessor(FrameProcessor):
    """
    Bridges Pipecat's real-time audio pipeline with a LangChain Deep Agent.

    The processor:
    1. Receives the completed user transcript from Pipecat.
    2. Sends it to the Deep Agent.
    3. Streams the generated text to the TTS processor.
    """

    def __init__(self, thread_id: str = "1"):
        super().__init__()

        # The thread ID separates conversation memory between sessions.
        self.config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # Retains conversation history in memory while the process is running.
        self.agent = build_deep_agent(checkpointer=InMemorySaver())

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ):
        """
        Process frames from the Pipecat pipeline.

        LLMContextFrame contains the completed conversational context after the
        user finishes speaking. All other frames pass through unchanged.
        """
        await super().process_frame(frame, direction)

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        # Find the latest user transcript. Universal-context messages are
        # normally plain dicts, but LLM-specific message objects can appear, so
        # read the role defensively instead of assuming a dict.
        user_text = ""

        for message in reversed(frame.context.messages):
            role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
            if role == "user":
                raw = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
                user_text = extract_text(raw).strip()
                break

        if not user_text:
            return

        # Tell downstream processors that generation is beginning.
        await self.push_frame(LLMFullResponseStartFrame())

        try:
            # Stream the agent's answer so TTS can start speaking before the
            # full response is finished.
            async for event in self.agent.astream_events(
                {"messages": [HumanMessage(content=user_text)]},
                config=self.config,
                version="v2",
            ):
                if event["event"] != "on_chat_model_stream":
                    continue

                text = extract_text(event["data"]["chunk"].content)

                if text:
                    await self.push_frame(TextFrame(text=text))

        except Exception:
            # Surface the failure in the server log; a silent except here is
            # what made the original problem so hard to see.
            logger.exception("Deep Agent failed while generating a response")
            await self.push_frame(
                TextFrame(text="Sorry, I ran into an error handling that request.")
            )

        finally:
            # TTS needs the end frame to flush its final sentence, and the
            # assistant aggregator needs it to close the turn. Both must happen
            # even when generation raised.
            await self.push_frame(LLMFullResponseEndFrame())
