bash -lc 'cd /home/LTX-2/packages/ltx-trainer && python -m ltx_trainer.direction_discovery run-cycle-consistency \
  /home/LTX-2/downloads/15510000_640_360_60fps.mp4 \
  /home/LTX-2/downloads/ltx-2.3-22b-dev.safetensors \
  /home/LTX-2/downloads/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "cinematic shot of the same scene" \
  --output-dir /home/LTX-2/outputs/cycle_consistency_$(date +%Y%m%d_%H%M%S) \
  --num-inference-steps 30 \
  --guidance-scale 4.0 \
  --device cuda'