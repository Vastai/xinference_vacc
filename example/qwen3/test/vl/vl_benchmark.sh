#!/bin/bash
set -euo pipefail

# ===================== 配置区域（可根据需求修改）=====================
MODEL_NAME="Qwen3-VL-Instruct"
PORT=9999
OUTPUT_CSV="qwen3_vl_results_xinf.csv"

# 测试参数列表
CONCURRENCY_LIST=(2 4)
RESOLUTION_LIST=("1280x720" "1920x1080" "2560x1440")
RANDOM_INPUT_LEN_LIST=(128 256 512 1024)
RANDOM_OUTPUT_LEN_LIST=(1024)

# ✅ 新版：支持 [分辨率:并发数:请求数] 多维配置
REQUESTS_CONFIG=(
    "1280x720:2:20"
    "1280x720:4:20"
    "1920x1080:2:10"
    "1920x1080:4:12"
    "2560x1440:2:8"
    "2560x1440:4:12"
    "default:20"   # 兜底：未匹配时用此值
)

TEST_SCRIPT="vl_benchmark.py"

# ===================== 内部函数 =====================
# 🔍 获取指定 (分辨率, 并发数) 对应的请求数
get_requests_count() {
    local resolution=$1
    local concurrency=$2
    local default_requests=20  # 默认兜底值（后续会被 config 覆盖）

    # 先查找精确匹配：resolution:concurrency
    for config in "${REQUESTS_CONFIG[@]}"; do
        IFS=":" read -r res conc cnt <<< "$config"
        if [[ "$res" == "$resolution" ]] && [[ "$conc" == "$concurrency" ]]; then
            echo "$cnt"
            return
        fi
        # 记录 default
        if [[ "$res" == "default" ]]; then
            default_requests=$cnt
        fi
    done

    # 未找到精确匹配 → 尝试仅按分辨率匹配（兼容旧逻辑，可选）
    for config in "${REQUESTS_CONFIG[@]}"; do
        IFS=":" read -r res conc cnt <<< "$config"
        if [[ "$res" == "$resolution" ]] && [[ "$conc" != "default" ]] && [[ "$conc" != *[!0-9]* ]]; then
            # 这是旧格式 "res:cnt"（无并发字段），按旧逻辑处理
            echo "$cnt"
            return
        fi
    done

    # 最终兜底
    echo "$default_requests"
}

print_separator() {
    echo "=============================================================="
}

# ===================== 主流程 =====================
if [[ ! -f "$TEST_SCRIPT" ]]; then
    echo "❌ 错误：测试脚本 $TEST_SCRIPT 不存在！"
    exit 1
fi

if [[ -f "$OUTPUT_CSV" ]]; then
    echo "ℹ️  清理旧的输出文件：$OUTPUT_CSV"
    rm -f "$OUTPUT_CSV"
fi

TOTAL_COMBINATIONS=$(( ${#CONCURRENCY_LIST[@]} * ${#RESOLUTION_LIST[@]} * ${#RANDOM_INPUT_LEN_LIST[@]} * ${#RANDOM_OUTPUT_LEN_LIST[@]} ))
CURRENT_COMBINATION=0

echo "🚀 开始Qwen3-VL批量性能测试"
print_separator
echo "模型名称：$MODEL_NAME"
echo "端口：$PORT"
echo "测试组合总数：$TOTAL_COMBINATIONS"
echo "并发数列表：${CONCURRENCY_LIST[*]}"
echo "分辨率列表：${RESOLUTION_LIST[*]}"
echo "输入Token列表：${RANDOM_INPUT_LEN_LIST[*]}"
echo "输出Token列表：${RANDOM_OUTPUT_LEN_LIST[*]}"
print_separator

# 👇 遍历：先分辨率 → 再并发（以便按需配置 requests）
for resolution in "${RESOLUTION_LIST[@]}"; do
    for concurrency in "${CONCURRENCY_LIST[@]}"; do
        # ✅ 关键：动态获取当前 (res, conc) 对应请求数
        requests=$(get_requests_count "$resolution" "$concurrency")
        
        for input_len in "${RANDOM_INPUT_LEN_LIST[@]}"; do
            for output_len in "${RANDOM_OUTPUT_LEN_LIST[@]}"; do
                CURRENT_COMBINATION=$((CURRENT_COMBINATION + 1))
                
                print_separator
                echo "📝 测试组合 [$CURRENT_COMBINATION/$TOTAL_COMBINATIONS]"
                echo "分辨率：$resolution | 并发数：$concurrency"
                echo "输入Token：$input_len | 输出Token：$output_len | 请求数：$requests"
                print_separator
                
                # 运行测试
                python "$TEST_SCRIPT" \
                    --model-name "$MODEL_NAME" \
                    --port "$PORT" \
                    --resolution "$resolution" \
                    --requests "$requests" \
                    --concurrency "$concurrency" \
                    --random-input-len "$input_len" \
                    --random-output-len "$output_len" \
                    --output "$OUTPUT_CSV"
                
                if [[ $? -eq 0 ]]; then
                    echo "✅ 测试组合完成：$resolution | $concurrency | $input_len | $output_len"
                else
                    echo "❌ 测试组合失败：$resolution | $concurrency | $input_len | $output_len"
                    # exit 1  # 可选：失败是否中止
                fi
                
                sleep 2
            done
        done
    done
done

print_separator
echo "🎉 所有测试组合执行完成！"
echo "📊 测试结果已保存至：$OUTPUT_CSV"
print_separator