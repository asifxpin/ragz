# ragz

A bilingual (English + Arabic) voice agent. Speech comes in over WebRTC, a
LangChain deep agent answers it with retrieval over a local knowledge base, and
the reply is streamed back as speech.

## How it fits together

Three files, each with one job:

| File | Responsibility |
| --- | --- |
| `main.py` | The Pipecat pipeline and entry point: WebRTC transport, ElevenLabs STT, Silero VAD, ElevenLabs TTS. |
| `deep_agent.py` | `DeepAgentProcessor`, the Pipecat processor that hands each finished user turn to a LangChain deep agent and streams the tokens onward to TTS. |
| `rags.py` | Bilingual retrieval: Arabic normalization, Arabic-aware chunking, OpenAI embeddings, FAISS index, and the `retrieve` tool the agent calls. |

Frame flow through the pipeline:

```
audio in -> STT -> user turn aggregator -> deep agent -> TTS -> audio out
```

`demo.ipynb` is a scratchpad for trying pieces out interactively.

## Setup

Requires Python 3.12 or newer.

```bash
uv sync
```

Then copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | yes | Used by the deep agent and by `text-embedding-3-large`. |
| `ELEVENLABS_API_KEY` | yes | Scribe v2 Realtime for STT, Flash v2.5 for TTS. |
| `OPENAI_MODEL` | no | Chat model. Defaults to `gpt-5.5`. |
| `ELEVENLABS_VOICE_ID` | no | Defaults to `21m00Tcm4TlvDq8ikWAM`. |

## Run

```bash
uv run python main.py
```

Open <http://localhost:7860>. That redirects to Pipecat's prebuilt client, which
asks for microphone access and then connects over WebRTC.

## Knowledge base

`rags.py` indexes every `.txt` file under `./files` at startup, searching
subfolders, so `./files/_samples` is picked up as well. Chunks are stored
normalized for reliable matching while the original wording is kept in metadata
so the agent reads natural text.

Drop in more `.txt` files and restart to extend the knowledge base. The index is
built in memory and is never written to disk.

Retrieval on its own, without starting the voice pipeline:

```bash
uv run python rags.py
uv run python rags.py "What did Rahul order?" "ماذا طلب راهول؟"
```

## Bilingual handling

Arabic needs work that English does not, so `rags.py` normalizes it before
embedding: diacritics are stripped, and alef, alef maksura and teh marbuta
variants are collapsed. Chunking splits on Arabic punctuation (`؟ ؛ ،`) as well
as English, which keeps sentences intact instead of cutting them at spaces.

`text-embedding-3-large` places both languages in one vector space, so an Arabic
question can retrieve an English chunk and vice versa. STT language detection is
pinned to English and Arabic; left open across all 90+ supported languages,
short or noisy speech gets misread as something unrelated.
