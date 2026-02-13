FROM madiator2011/better-pytorch:cuda12.4-torch2.6.0

RUN apt-get update && apt-get install -y libsndfile1 git git-lfs && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/index-tts/index-tts.git /indextts && \
    cd /indextts && git lfs pull && \
    pip install -e . --no-cache-dir && \
    pip install runpod --no-cache-dir

RUN huggingface-cli download IndexTeam/IndexTTS-2 --local-dir=/indextts/checkpoints

RUN cd /indextts && python -c "\
from indextts.infer_v2 import IndexTTS2; \
IndexTTS2(cfg_path='checkpoints/config.yaml', model_dir='checkpoints', use_fp16=True, device='cpu')" || true

COPY handler.py /handler.py

WORKDIR /indextts

CMD ["python", "-u", "/handler.py"]
