#!/usr/bin/env bash
# Few-shot SER with GeLDA in Z^(0) for one (SFM, target language, fold): Stages 1-3.
#
#   bash recipes/few_shot.sh whisper-large-v3 greek 1
#
# Stage 1 and Stage 2 are trained once per (SFM, fold) on the full multilingual
# set D and shared by all target languages. Every stage is run by default. A
# stage can be skipped by using the released checkpoint in checkpoints/ instead:
#
#   SKIP_STAGE1=1   use the released Stage-1 adapter
#   SKIP_STAGE2=1   use the released diffusion model (sampling is still done)
set -euo pipefail

SFM=${1:-whisper-large-v3}
LANG_=${2:-greek}
FOLD=${3:-1}
SKIP_STAGE1=${SKIP_STAGE1:-0}
SKIP_STAGE2=${SKIP_STAGE2:-0}
NUM_AUG=${NUM_AUG:-500}
REPETITIONS=${REPETITIONS:-"0 1 2 3 4"}

MANIFESTS=manifests/full
FEATURES=features/${SFM}
RELEASED=checkpoints/few_shot/${SFM}/fold${FOLD}
SHARED=runs/few_shot/${SFM}/fold${FOLD}
RUN=${SHARED}/${LANG_}
mkdir -p "${RUN}"

# ---- Stage 1: SER adapter on the full set D (shared by all target languages)
if [[ "${SKIP_STAGE1}" == "1" ]]; then
    ADAPTER=${RELEASED}/ser_adapter.safetensors
else
    [[ -f "${SHARED}/stage1/adapter.safetensors" ]] || python scripts/train_ser.py \
        --manifest-dir "${MANIFESTS}" --fold "${FOLD}" --features "${FEATURES}" --out "${SHARED}/stage1"
    ADAPTER=${SHARED}/stage1/adapter.safetensors
fi

# ---- Stage 2: latent diffusion model on D (shared by all target languages)
if [[ "${SKIP_STAGE2}" == "1" ]]; then
    DIFFUSION=${RELEASED}/diffusion_z0.safetensors
else
    [[ -f "${SHARED}/diffusion_z0/diffusion.safetensors" ]] || python scripts/train_diffusion.py \
        --config "configs/diffusion/${SFM}_z0.json" --manifest-dir "${MANIFESTS}" --fold "${FOLD}" \
        --latents "${FEATURES}" --out "${SHARED}/diffusion_z0"
    DIFFUSION=${SHARED}/diffusion_z0/diffusion.safetensors
fi

N_REPS=$(wc -w <<< "${REPETITIONS}")
python scripts/generate.py --checkpoint "${DIFFUSION}" --manifest-dir "${MANIFESTS}" --fold "${FOLD}" \
    --latents "${FEATURES}" --language "${LANG_}" --num-samples $((NUM_AUG * N_REPS)) \
    --out "${RUN}/generated_z0"

# ---- Stage 3: fine-tune on the labelled target-language data + generated latents.
# --num-aug 0 gives the "Fine-tuned" baseline.
for REP in ${REPETITIONS}; do
    python scripts/finetune_ser.py --setting few_shot --language "${LANG_}" --fold "${FOLD}" \
        --layer 0 --manifest-dir "${MANIFESTS}" --latents "${FEATURES}" --adapter "${ADAPTER}" \
        --generated "${RUN}/generated_z0" --num-aug "${NUM_AUG}" --repetition "${REP}" \
        --out "${RUN}/stage3_z0_rep${REP}"
done
