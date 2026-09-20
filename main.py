import os
import sys
from urllib.parse import quote

# Windows consoles default to cp1252, which cannot encode the Unicode box-drawing
# and glyph characters Pipecat prints in its startup banner. Force UTF-8 on the
# streams before importing Pipecat so those prints don't raise UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# Import the custom Deep Agent processor from deep_agent.py
from deep_agent import DeepAgentProcessor

# Pipecat Core & Audio components
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

# Load environment variables (.env) for API keys
load_dotenv()


# ============================================================================
# 0. BILINGUAL STT SERVICE
# ============================================================================
class BilingualSTTService(ElevenLabsRealtimeSTTService):
    """ElevenLabs realtime STT with language identification pinned to a set.

    Pipecat 1.10 does not expose ElevenLabs' ``secondary_languages`` query
    parameter. Without it, detection ranges across 90+ languages, and short or
    noisy speech gets misread as an unrelated language (Polish, Portuguese),
    which the model then transcribes *as* that language. Supplying the set we
    actually support keeps identification inside English and Arabic.

    Only the URI is extended here, so Pipecat keeps ownership of building
    every other query parameter.
    """

    def __init__(self, *, secondary_languages: tuple[str, ...] = (), **kwargs):
        super().__init__(**kwargs)
        self._secondary_languages = secondary_languages

    async def _websocket_connect(self, uri: str, **kwargs):
        for language in self._secondary_languages:
            uri += f"&secondary_languages={quote(language)}"
        return await super()._websocket_connect(uri, **kwargs)


# ============================================================================
# 1. TRANSPORT CONFIGURATION
# ============================================================================
# Configure WebRTC transport parameters to enable two-way audio (input and output)
transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


# ============================================================================
# 2. MAIN BOT PIPELINE SETUP
# ============================================================================
async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """
    Initializes services, constructs the frame-processing pipeline, and runs the agent.
    """
    # WorkerRunner coordinates agent lifecycle and clean shutdown
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    # 1. Speech-to-Text (STT): ElevenLabs Scribe v2 Realtime.
    # language + secondary_languages restrict identification to English and
    # Arabic instead of guessing across every supported language, and
    # filter_background_audio drops ambient noise and echo before transcription.
    # Codes are ISO-639-3, matching ElevenLabs' own language listing.
    elevenlabs_api_key = os.environ["ELEVENLABS_API_KEY"]
    stt = BilingualSTTService(
        api_key=elevenlabs_api_key,
        include_language_detection=True,
        secondary_languages=("ara",),
        settings=ElevenLabsRealtimeSTTService.Settings(
            model="scribe_v2_realtime",
            language="eng",
            filter_background_audio=True,
        ),
    )

    # 2. Intelligence: Custom LangChain Deep Agent Processor with Memory & Weather Tool
    deep_agent = DeepAgentProcessor(thread_id="customer_session_1")

    # 3. Text-to-Speech (TTS): ElevenLabs multilingual WebSocket streaming.
    # Flash v2.5 supports both English and Arabic. Omitting a language code lets
    # ElevenLabs infer pronunciation from each response, including mixed text.
    tts = ElevenLabsTTSService(
        api_key=elevenlabs_api_key,
        settings=ElevenLabsTTSService.Settings(
            voice=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            model="eleven_flash_v2_5",
            language=None,
        ),
    )

    # 4. Context & Voice Activity Detection (VAD):
    # Silero VAD detects when the user starts/stops speaking to manage conversation turns.
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # 5. Pipeline Assembly:
    # Frame flow: Audio In -> Transcribe -> Turn Aggregation -> Agent Reasoning -> Speech Synthesis -> Audio Out
    pipeline = Pipeline(
        [
            transport.input(),       # 1. Receives raw audio chunks from user's browser
            stt,                     # 2. Converts audio chunks to text
            aggregators.user(),      # 3. Groups user text into complete sentences (emits LLMContextFrame)
            deep_agent,              # 4. Deep Agent processes message, invokes tools, streams response tokens
            tts,                     # 5. Converts streamed text tokens back into voice audio
            transport.output(),      # 6. Sends synthesized audio back to user's browser
            aggregators.assistant(), # 7. Records assistant response into conversation context
        ]
    )

    # 6. Pipeline Worker: Manages pipeline execution and metrics
    agent = PipelineWorker(
        pipeline,
        name="assistant",
        params=PipelineParams(enable_metrics=True),
    )

    # Trigger initial greeting frame as soon as the client browser connects and is ready
    @agent.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await agent.queue_frames([LLMRunFrame()])

    # Cleanly stop the pipeline when the user disconnects / closes the browser tab
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await runner.cancel()

    # Register worker with the runner and start processing
    await runner.add_workers(agent)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Entry point invoked by Pipecat CLI / runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


# ============================================================================
# 3. SCRIPT ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    from pipecat.runner.run import main

    main()