python -m ltx_trainer.direction_discovery run-direction-discovery \
  /home/LTX-2/downloads/15510000_640_360_60fps.mp4 \
  --checkpoint-path /home/LTX-2/downloads/ltx-2.3-22b-dev.safetensors \
  --text-encoder-path /home/LTX-2/downloads/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "cinematic shot of the same scene" \
  --method random \
  --num-directions 1 \
  --single-alpha 8.0 \
  --direction-target-rms 0.25 \
  --edit-after-diffusion \
  --output-dir /home/LTX-2/outputs/direction_discovery_one_dir_one_alpha_$(date +%Y%m%d_%H%M%S)




  # python -m ltx_trainer.direction_discovery run-direction-discovery \
  # /home/LTX-2/downloads/15510000_640_360_60fps.mp4 \
  # --checkpoint-path /home/LTX-2/downloads/ltx-2.3-22b-dev.safetensors \
  # --text-encoder-path /home/LTX-2/downloads/gemma-3-12b-it-qat-q4_0-unquantized \
  # --prompt "cinematic shot of the same scene" \
  # --method difference \
  # --num-directions 1 \
  # --grayscale-difference-direction \
  # --save-grayscale-reference \
  # --edit-after-diffusion \
  # --single-alpha -1.0 \
  # --direction-target-rms 0.1 \
  # --seed 123 \
  # --diffusion-noise-scale 0.2 \
  # --transfer-video-path /home/LTX-2/downloads/3371825-sd_426_240_24fps.mp4 \
  # --transfer-alpha -1.0 \
  # --output-dir /home/LTX-2/outputs/direction_discovery_bw_transfer_$(date +%Y%m%d_%H%M%S)

  python -m ltx_trainer.direction_discovery run-transfer-saved-direction \
  /home/LTX-2/outputs/direction_discovery_zoom_20260409_112055/directions/direction_0.pt \
  /home/LTX-2/downloads/15510000_640_360_60fps.mp4 \
  /home/LTX-2/downloads/3371825-sd_426_240_24fps.mp4 \
  /home/LTX-2/downloads/8154896-sd_960_540_25fps.mp4 \
  /home/LTX-2/downloads/9464705-sd_960_506_25fps.mp4 \
  --checkpoint-path /home/LTX-2/downloads/ltx-2.3-22b-dev.safetensors \
  --text-encoder-path /home/LTX-2/downloads/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "cinematic shot of the same scene" \
  --alpha -1.0 \
  --direction-target-rms 0.1 \
  --seed 123 \
  --diffusion-noise-scale 0.2 \
  --edit-after-diffusion \
  --output-dir /home/LTX-2/outputs/direction_transfer_multi_$(date +%Y%m%d_%H%M%S)