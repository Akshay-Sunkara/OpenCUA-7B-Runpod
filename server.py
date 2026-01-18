from vllm import vLLMEngine, AsyncEngineArgs
from fastapi import FastAPI
from pydantic import BaseModel

class RequestBody(BaseModel):
    prompt: str
    max_tokens: int = 200

app = FastAPI()

engine_args = AsyncEngineArgs(
    model="xlangai/opencua-7b",
    trust_remote_code=True,
    runner="auto",
    dtype="auto",
    gpu_memory_utilization=0.95
)
vllm_engine = vLLMEngine(engine_args)

@app.post("/generate")
async def generate(request: RequestBody):
    result = vllm_engine.generate(request.prompt, max_output_tokens=request.max_tokens)
    output_text = "\n".join([r.text for r in result])
    return {"output": output_text}
