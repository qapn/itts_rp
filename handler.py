import runpod
import traceback

model = None
INIT_ERROR = None


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


try:
    load_model()
except Exception:
    INIT_ERROR = traceback.format_exc()
    print(f"[init] FAILED:\n{INIT_ERROR}", flush=True)


def handler(job):
    if INIT_ERROR:
        return {'error': f'Model failed to load:\n{INIT_ERROR}'}

    import base64
    import io
    import os
    import re
    import tempfile

    import soundfile as sf

    inp = job['input']

    text = inp.get('text', '')
    if not text:
        return {'error': 'text is required'}

    text = re.sub(r'\[[^\]]+\]', '', text).strip()
    if not text:
        return {'error': 'text is empty after tag stripping'}

    ref_audio_b64 = inp.get('reference_audio_base64')
    ref_path = None
    out_path = None

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

        out_fd = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        out_path = out_fd.name
        out_fd.close()

        emo_vector = inp.get('emo_vector')
        emo_alpha = float(inp.get('emo_alpha', 0.7))

        gen_kwargs = {}
        for key in ('temperature', 'top_k', 'top_p', 'num_beams', 'max_mel_tokens', 'seed'):
            if key in inp:
                val = inp[key]
                gen_kwargs[key] = int(val) if key in ('top_k', 'num_beams', 'max_mel_tokens', 'seed') else float(val)

        model.infer(
            spk_audio_prompt=ref_path,
            text=text,
            output_path=out_path,
            emo_vector=emo_vector,
            emo_alpha=emo_alpha,
            **gen_kwargs,
        )

        with open(out_path, 'rb') as f:
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
        if out_path and os.path.exists(out_path):
            os.unlink(out_path)


runpod.serverless.start({'handler': handler})
