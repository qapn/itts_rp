FROM madiator2011/better-pytorch:cuda12.4-torch2.6.0

RUN apt-get update && apt-get install -y libsndfile1 git && rm -rf /var/lib/apt/lists/*

RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/index-tts/index-tts.git /indextts

RUN cd /indextts && \
    pip install -e . --no-cache-dir && \
    pip install runpod torchmetrics librosa requests --no-cache-dir

WORKDIR /indextts

RUN huggingface-cli download IndexTeam/IndexTTS-2 --local-dir=checkpoints

ENV HF_HUB_CACHE=/indextts/checkpoints/hf_cache
RUN huggingface-cli download facebook/w2v-bert-2.0
RUN huggingface-cli download amphion/MaskGCT
RUN huggingface-cli download funasr/campplus
RUN huggingface-cli download nvidia/bigvgan_v2_22khz_80band_256x
RUN huggingface-cli download facebook/wav2vec2-base

RUN python -c "import torch; torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True)"
RUN python -c "from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment; import torch; m = NonIntrusiveSpeechQualityAssessment(16000); m(torch.randn(16000))"

ENV HF_HUB_OFFLINE=1

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
