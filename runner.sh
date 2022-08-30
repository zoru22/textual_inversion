#!/usr/bin/env


function runInference()
{
  num_samples=$1
  num_iter=$2
  scale=$3
  ddim_steps=$4
  full_embedding_path=$5
  embedding_base=$(basename "${full_embedding_path%.pt}")
  model_ckpt=$6
  prompt=$7

  if [ "$embedding_base" == "embeddings" ] || [ "$embedding_base" == "" ]; then
    echo "cant run inference b/c embedding_base == '$embedding_base'"
    return 1
  fi

  python scripts/stable_txt2img.py --ddim_eta 0 \
  --n_samples "$num_samples" \
  --n_iter "$num_iter" \
  --scale "$scale" \
  --ddim_steps "$ddim_steps" \
  --embedding_file "$full_embedding_path" \
  --ckpt "$model_ckpt" \
  --prompt "$prompt" \
  --seed 246 \
  --outdir "outputs/tf/$embedding_base"

  echo "ran inference!"

  return $?
}

# I use stable diffusion's v1.3 ckpt
modelCkpt="models/ldm/stable-diffusion/model.ckpt"
prompt="8k * canon transformation high quality realistic kodak picture of * into pool toy shark synthetic"
embedding_ckpt_folder="logs/hoop/2022-08-30T11-37-04_hoop_tf_one/checkpoints"

for f in "./$embedding_ckpt_folder/"*.pt; do
  g=$(basename "${f%.pt}");
  echo "$g"
  if [ "$g" == "embeddings" ]; then
    continue
  fi
#  apparently 200 steps is ideal?
  runInference 3 2 10 100 "$f" "$modelCkpt" "$prompt"
  if [ $? != 0 ]; then
    echo "failed, path: $f"
    exit $?
  fi
done

#echo "ran inference on $totalRuns total runs with prompt: '$prompt'!1!"
exit $?

