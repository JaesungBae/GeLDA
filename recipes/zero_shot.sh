#!/usr/bin/env bash
# Zero-shot SER with GeLDA for one (SFM, target language, fold, latent layer): Stages 1-3.
#
#   bash recipes/zero_shot.sh whisper-large-v3 french 1 0
#
# Assumes manifests/ and features/<sfm>/ exist (see README). Every stage is run
# by default. A stage can be skipped by using the released checkpoint in
# checkpoints/ instead:
#
#   SKIP_STAGE1=1   use the released Stage-1 adapter
#   SKIP_STAGE2=1   use the released diffusion model (sampling is still done)
#
# The released diffusion models in Z(1)/Z(2) were trained on latents of the
# released Stage-1 adapter, so for layers 1 and 2 SKIP_STAGE2=1 needs SKIP_STAGE1=1.
set -euo pipefail

SFM=${1:-whisper-large-v3}
LANG_=${2:-french}
FOLD=${3:-1}
LAYER=${4:-0}
SKIP_STAGE1=${SKIP_STAGE1:-0}
SKIP_STAGE2=${SKIP_STAGE2:-0}
NUM_AUG=${NUM_AUG:-200}
REPETITIONS=${REPETITIONS:-"0 1 2 3 4"}

if [[ "${LAYER}" != "0" && "${SKIP_STAGE2}" == "1" && "${SKIP_STAGE1}" != "1" ]]; then
    echo "SKIP_STAGE2=1 with layer ${LAYER} requires SKIP_STAGE1=1" >&2
    exit 1
fi

MANIFESTS=manifests/zero_shot_${LANG_}
FEATURES=features/${SFM}
RELEASED=checkpoints/zero_shot/${SFM}/${LANG_}/fold${FOLD}
RUN=runs/zero_shot/${SFM}/${LANG_}/fold${FOLD}
mkdir -p "${RUN}"

# ---- Stage 1: language-agnostic SER adapter on D^(\k) U D_NEU^(k)
if [[ "${SKIP_STAGE1}" == "1" ]]; then
    ADAPTER=${RELEASED}/ser_adapter.safetensors
else
    python scripts/train_ser.py --manifest-dir "${MANIFESTS}" --fold "${FOLD}" \
        --features "${FEATURES}" --out "${RUN}/stage1"
    ADAPTER=${RUN}/stage1/adapter.safetensors
fi

# Latents of Z^(LAYER)
if [[ "${LAYER}" == "0" ]]; then
    LATENTS=${FEATURES}
else
    python scripts/extract_latents.py --adapter "${ADAPTER}" --manifest-dir "${MANIFESTS}" \
        --fold "${FOLD}" --features "${FEATURES}" --out "${RUN}/latents"
    LATENTS=${RUN}/latents/z${LAYER}
fi

# ---- Stage 2: latent diffusion model conditioned on emotion + language reference
if [[ "${SKIP_STAGE2}" == "1" ]]; then
    DIFFUSION=${RELEASED}/diffusion_z${LAYER}.safetensors
else
    python scripts/train_diffusion.py --config "configs/diffusion/${SFM}_z${LAYER}.json" \
        --manifest-dir "${MANIFESTS}" --fold "${FOLD}" --latents "${LATENTS}" \
        --out "${RUN}/diffusion_z${LAYER}"
    DIFFUSION=${RUN}/diffusion_z${LAYER}/diffusion.safetensors
fi

N_REPS=$(wc -w <<< "${REPETITIONS}")
python scripts/generate.py --checkpoint "${DIFFUSION}" --manifest-dir "${MANIFESTS}" --fold "${FOLD}" \
    --latents "${LATENTS}" --language "${LANG_}" --num-samples $((NUM_AUG * N_REPS)) \
    --out "${RUN}/generated_z${LAYER}"

# ---- Stage 3: fine-tune the layers above Z^(LAYER) on target NEUTRAL data + generated latents
for REP in ${REPETITIONS}; do
    python scripts/finetune_ser.py --setting zero_shot --language "${LANG_}" --fold "${FOLD}" \
        --layer "${LAYER}" --manifest-dir "${MANIFESTS}" --latents "${LATENTS}" \
        --adapter "${ADAPTER}" --generated "${RUN}/generated_z${LAYER}" --num-aug "${NUM_AUG}" \
        --repetition "${REP}" --out "${RUN}/stage3_z${LAYER}_rep${REP}"
done
