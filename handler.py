import runpod
import traceback

model = None
utmos_model = None
nisqa_model = None
mel_fn = None
resample_fn = None
INIT_ERROR = None

SEED_STRIDE = 10000
W = (0.40, 0.20, 0.25, 0.15)


def load_model():
    global model

    print("[init] Importing IndexTTS2...", flush=True)
    from indextts.infer_v2 import IndexTTS2

    print("[init] Loading model...", flush=True)
    model = IndexTTS2(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_fp16=True,
    )

    print("[init] Model ready.", flush=True)


def load_scorers():
    global utmos_model, nisqa_model, mel_fn, resample_fn
    import torch
    import torchaudio

    print("[init] Loading scorers...", flush=True)

    utmos_model = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    )
    utmos_model.eval()
    if torch.cuda.is_available():
        utmos_model = utmos_model.cuda()

    from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment
    nisqa_model = NonIntrusiveSpeechQualityAssessment(16000)

    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80
    )
    if torch.cuda.is_available():
        mel_fn = mel_fn.cuda()

    resample_fn = torchaudio.transforms.Resample(22050, 16000)

    print("[init] Scorers ready.", flush=True)


try:
    load_model()
    load_scorers()
except Exception:
    INIT_ERROR = traceback.format_exc()
    print(f"[init] FAILED:\n{INIT_ERROR}", flush=True)


def _utmos(audio_np):
    import torch
    wav = resample_fn(torch.from_numpy(audio_np).unsqueeze(0))
    if torch.cuda.is_available():
        wav = wav.cuda()
    with torch.no_grad():
        return float(utmos_model(wav, sr=16000).mean().item())


def _nisqa(audio_np):
    import torch
    wav = resample_fn(torch.from_numpy(audio_np).float())
    return float(nisqa_model(wav, 16000)[0].item())


def _mel_vec(audio_np):
    import torch
    wav = torch.from_numpy(audio_np).unsqueeze(0)
    if torch.cuda.is_available():
        wav = wav.cuda()
    with torch.no_grad():
        mel = mel_fn(wav)
    return mel.squeeze(0).mean(dim=-1)


def _hnr(audio_np, sr=22050):
    import numpy as np
    frame_size = 2048
    hop = 512
    lag_lo = sr // 350
    lag_hi = sr // 70
    n_frames = (len(audio_np) - frame_size) // hop + 1
    if n_frames < 1:
        return 0.0
    vals = []
    for i in range(n_frames):
        frame = audio_np[i * hop: i * hop + frame_size]
        if np.sqrt(np.mean(frame ** 2)) < 0.01:
            continue
        w = frame * np.hanning(frame_size)
        pwr = np.abs(np.fft.rfft(w)) ** 2
        acf = np.fft.irfft(pwr)
        acf = acf / (acf[0] + 1e-10)
        region = acf[lag_lo:lag_hi + 1]
        if len(region) == 0:
            continue
        pk = float(np.max(region))
        if 0 < pk < 1:
            vals.append(10 * np.log10(pk / (1 - pk + 1e-10)))
    return float(np.mean(vals)) if vals else 0.0


def _pick(candidates):
    import numpy as np
    import torch

    n = len(candidates)
    scores = np.zeros((n, 4))

    for i, c in enumerate(candidates):
        scores[i, 0] = _utmos(c)
        scores[i, 1] = _nisqa(c)
        scores[i, 3] = _hnr(c)

    vecs = [_mel_vec(c) for c in candidates]
    for i in range(n):
        sims = []
        for j in range(n):
            if i == j:
                continue
            sim = torch.nn.functional.cosine_similarity(
                vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)
            )
            sims.append(float(sim.item()))
        scores[i, 2] = sum(sims) / len(sims)

    for col in range(4):
        lo, hi = scores[:, col].min(), scores[:, col].max()
        if hi > lo:
            scores[:, col] = (scores[:, col] - lo) / (hi - lo)
        else:
            scores[:, col] = 0.5

    combined = scores @ np.array(W)
    return int(np.argmax(combined))


def handler(job):
    if INIT_ERROR:
        return {'error': f'Model failed to load:\n{INIT_ERROR}'}

    import base64
    import io
    import os
    import random
    import re
    import tempfile

    import numpy as np
    import soundfile as sf
    import torch

    inp = job['input']

    text = inp.get('text', '')
    if not text:
        return {'error': 'text is required'}

    text = re.sub(r'\[[^\]]+\]', '', text).strip()
    if not text:
        return {'error': 'text is empty after tag stripping'}

    ref_audio_b64 = inp.get('reference_audio_base64')
    ref_path = None
    out_paths = []

    try:
        if ref_audio_b64:
            raw = base64.b64decode(ref_audio_b64)
            audio_data, sr = sf.read(io.BytesIO(raw), dtype='float32')

            max_samples = sr * 15
            if len(audio_data) > max_samples:
                audio_data = audio_data[:max_samples]

            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(tmp.name, audio_data, sr)
            tmp.close()
            ref_path = tmp.name

        emo_vector = inp.get('emo_vector')
        emo_alpha = float(inp.get('emo_alpha', 0.7))

        base_seed = int(inp.get('seed', 0))
        if base_seed == 0:
            base_seed = random.randint(1, 999999)

        gen_kwargs = {}
        _INT_KEYS = ('top_k', 'num_beams', 'max_mel_tokens')
        _BOOL_KEYS = ('do_sample',)
        for key in ('temperature', 'top_k', 'top_p', 'num_beams', 'max_mel_tokens',
                     'do_sample', 'repetition_penalty', 'length_penalty'):
            if key in inp:
                val = inp[key]
                if key in _INT_KEYS:
                    gen_kwargs[key] = int(val)
                elif key in _BOOL_KEYS:
                    gen_kwargs[key] = bool(val)
                else:
                    gen_kwargs[key] = float(val)

        seeds = [base_seed + i * SEED_STRIDE for i in range(3)]
        candidates = []

        for s in seeds:
            torch.manual_seed(s)
            random.seed(s)

            out_fd = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            out_path = out_fd.name
            out_fd.close()
            out_paths.append(out_path)

            model.infer(
                spk_audio_prompt=ref_path,
                text=text,
                output_path=out_path,
                emo_vector=emo_vector,
                emo_alpha=emo_alpha,
                **gen_kwargs,
            )

            audio_np, _ = sf.read(out_path, dtype='float32')
            candidates.append(audio_np)

        best = _pick(candidates)

        with open(out_paths[best], 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')

        return {
            'audio_base64': audio_base64,
            'sample_rate': 22050,
            'format': 'wav',
        }

    except Exception as e:
        return {'error': str(e)}

    finally:
        if ref_path:
            os.unlink(ref_path)
        for p in out_paths:
            if os.path.exists(p):
                os.unlink(p)


runpod.serverless.start({'handler': handler})
