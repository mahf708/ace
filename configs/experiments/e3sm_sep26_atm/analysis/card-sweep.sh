#!/bin/bash
# One-node memory + step-cost sweep for the atmosphere stepper, on whatever
# card the allocation gave us. Each variant runs until it has logged 7 step
# lines (70 batches) or hits a deadline; peak GPU memory comes from nvidia-smi.
set -u
REPO=/pscratch/sd/m/mahf708/ace-sep26
SP="$1"; TAG="$2"; CFG="$SP/smoke-atm.yaml"
TORCHRUN=$REPO/.venv/bin/torchrun
OUTROOT=$PSCRATCH/sep26-memprobe/$TAG
RES=$SP/memprobe/$TAG
mkdir -p "$RES" "$OUTROOT"
export MASTER_ADDR=$(hostname); export MASTER_PORT=29601
export PYTHONUNBUFFERED=1
echo "### tag=$TAG host=$(hostname)"
echo "### card: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "### host mem: $(free -g | awk '/^Mem:/{print $2" GB"}')"

VARIANTS=(
"E01_M2_RF1|stepper_training.n_ensemble=2 stepper_training.n_forward_steps=1 stepper_training.optimize_last_step_only=true"
"M1_RF1_G0_expect_crash|stepper_training.n_ensemble=1 stepper_training.n_forward_steps=1 stepper_training.optimize_last_step_only=true"
"M3_RF1_G0_expect_crash|stepper_training.n_ensemble=3 stepper_training.n_forward_steps=1 stepper_training.optimize_last_step_only=true"
"M1_RF1_G1|stepper_training.n_ensemble=1 stepper_training.n_forward_steps=1 stepper_training.optimize_last_step_only=true stepper_training.loss.kwargs.crps_weight=1.0 stepper_training.loss.kwargs.energy_score_weight=0.0"
"M3_RF1_G1|stepper_training.n_ensemble=3 stepper_training.n_forward_steps=1 stepper_training.optimize_last_step_only=true stepper_training.loss.kwargs.crps_weight=1.0 stepper_training.loss.kwargs.energy_score_weight=0.0"
"M2_RF2_bothscored|stepper_training.n_ensemble=2 stepper_training.n_forward_steps=2 stepper_training.optimize_last_step_only=false"
"M2_RC2_lastonly|stepper_training.n_ensemble=2 stepper_training.n_forward_steps=2 stepper_training.optimize_last_step_only=true"
"M2_R20_lastonly|stepper_training.n_ensemble=2 stepper_training.n_forward_steps=20 stepper_training.optimize_last_step_only=true"
"M2_RF1_Q1_fdcrps|stepper_training.n_ensemble=2 stepper_training.n_forward_steps=1 stepper_training.optimize_last_step_only=true stepper_training.loss.kwargs.finite_difference_crps_weight=0.1 stepper_training.loss.kwargs.finite_difference_crps_levels=1"
"M2_RF1_Q2_fdcrps3|stepper_training.n_ensemble=2 stepper_training.n_forward_steps=1 stepper_training.optimize_last_step_only=true stepper_training.loss.kwargs.finite_difference_crps_weight=0.1 stepper_training.loss.kwargs.finite_difference_crps_levels=3"
"M2_RF1_N1_gaussian|stepper_training.n_ensemble=2 stepper_training.n_forward_steps=1 stepper_training.optimize_last_step_only=true stepper.step.config.builder.config.noise_type=gaussian"
"D1_M1_Z00_mse|stepper_training.n_ensemble=1 stepper_training.n_forward_steps=1 stepper_training.optimize_last_step_only=true stepper_training.loss.type=MSE stepper.step.config.builder.config.noise_embed_dim=0 stepper.step.config.builder.config.noise_type=gaussian"
)

for entry in "${VARIANTS[@]}"; do
  name="${entry%%|*}"; ovr="${entry#*|}"
  out=$OUTROOT/$name; rm -rf "$out"; mkdir -p "$out"
  log=$RES/$name.log; mem=$RES/$name.mem
  echo "=== $name  $(date +%T) ==="
  ( while :; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; sleep 3; done ) > "$mem" 2>/dev/null &
  sampler=$!
  $TORCHRUN --nnodes 1 --nproc_per_node 4 --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" -m fme.ace.train "$CFG" \
    --override experiment_dir="$out" log_train_every_n_batches=10 \
      train_loader.num_data_workers=8 train_loader.prefetch_factor=4 \
      $ovr > "$log" 2>&1 &
  pid=$!
  deadline=$(( $(date +%s) + 900 ))
  while kill -0 $pid 2>/dev/null; do
    n=$(grep -c 'Step [0-9]*:' "$log" 2>/dev/null) || n=0
    [ "${n:-0}" -ge 7 ] && break
    [ "$(date +%s)" -ge "$deadline" ] && { echo "  (deadline)"; break; }
    sleep 5
  done
  kill -TERM $pid 2>/dev/null; sleep 5
  pkill -f 'fme[.]ace[.]train' 2>/dev/null
  sleep 5; wait $pid 2>/dev/null; kill $sampler 2>/dev/null
  peak=$(awk -F', ' '{if($2+0>m)m=$2+0}END{print m+0}' "$mem")
  steps=$(grep -c 'Step [0-9]*:' "$log") || steps=0
  err=$(grep -m1 -oE 'NotImplementedError[^"]{0,80}|torch.OutOfMemoryError[^"]{0,60}|CUDA out of memory' "$log" | head -1)
  # A peak sampled from a run that never reached a training step is the
  # setup-and-model-build high-water mark, not a training peak, and it reads
  # 20-25% low.  Label it so nobody differences it against a real one: the
  # 20-step rollout produced 23,233 MiB on a card that trained and 17,471 on a
  # card that timed out during dataset construction, which looked like a
  # cross-card disagreement until the step count was checked.
  valid=$([ "${steps:-0}" -ge 3 ] && echo yes || echo "NO-not-a-training-peak")
  printf 'RESULT|%s|%s|peak_MiB=%s|steps=%s|valid=%s|%s\n' \
      "$TAG" "$name" "${peak:-NA}" "${steps:-0}" "$valid" "${err:-ok}"
  rm -rf "$out"
  sleep 3
done
echo "### sweep done $(date +%T)"
