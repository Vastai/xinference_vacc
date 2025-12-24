#python benchmark.py --port 9999 --model Qwen3-VL-Instruct --prompt-token-len 200 --image-width 512 --image-height 512 --concurrency 1,2,4 --iterations 2 --output vl_test_results.csv
python vl_benchmark.py \
  --model-name "Qwen3-VL-Instruct" \
  --port 9999 \
  --resolution "1280x720" \
  --requests 1 \
  --concurrency 1 \
  --random-input-len 128 \
  --random-output-len 1024 \
  --output qwen3_vl_results.csv 