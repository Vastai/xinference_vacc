#!/bin/bash
set -euo pipefail

# ===================== 配置区域（可根据需求修改）=====================
# 模型基础配置
MODEL_NAME="Qwen3-VL-Instruct"
PORT=9999
OUTPUT_CSV="qwen3_vl_results.csv"  # 最终输出的CSV文件名

# 测试参数列表（自定义需要测试的组合）
CONCURRENCY_LIST=(1 2 4)                # 并发数列表
RESOLUTION_LIST=("1280x720" "1920x1080" "2560x1440")  # 分辨率列表
RANDOM_INPUT_LEN_LIST=(128 256 512 1024) # 输入Token长度列表
RANDOM_OUTPUT_LEN_LIST=(1024)           # 输出Token长度列表（可扩展）

# 每个测试组合的请求数（可按分辨率自定义）
# 格式："分辨率:请求数"，未匹配的分辨率使用默认值
REQUESTS_CONFIG=(
    "1280x720:3"
    "1920x1080:3"
    "2560x1440:3"
    "default:20"  # 默认请求数
)

# Python测试脚本路径（根据实际路径修改）
TEST_SCRIPT="vl_benchmark.py"

# ===================== 内部函数 =====================
# 获取指定分辨率对应的请求数
get_requests_count() {
    local resolution=$1
    for config in "${REQUESTS_CONFIG[@]}"; do
        IFS=":" read -r res cnt <<< "$config"
        if [[ "$res" == "$resolution" ]]; then
            echo "$cnt"
            return
        fi
        if [[ "$res" == "default" ]]; then
            DEFAULT_REQUESTS=$cnt
        fi
    done
    echo "$DEFAULT_REQUESTS"
}

# 打印分隔线
print_separator() {
    echo "=============================================================="
}

# ===================== 主流程 =====================
# 第一步：检查脚本存在性
if [[ ! -f "$TEST_SCRIPT" ]]; then
    echo "❌ 错误：测试脚本 $TEST_SCRIPT 不存在！"
    exit 1
fi

# 第二步：清理旧的输出文件（可选，注释掉则追加模式）
if [[ -f "$OUTPUT_CSV" ]]; then
    echo "ℹ️  清理旧的输出文件：$OUTPUT_CSV"
    rm -f "$OUTPUT_CSV"
fi

# 第三步：计算总测试组合数
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

# 第四步：遍历所有测试组合
for resolution in "${RESOLUTION_LIST[@]}"; do
    # 获取当前分辨率对应的请求数
    requests=$(get_requests_count "$resolution")
    
    for concurrency in "${CONCURRENCY_LIST[@]}"; do
        for input_len in "${RANDOM_INPUT_LEN_LIST[@]}"; do
            for output_len in "${RANDOM_OUTPUT_LEN_LIST[@]}"; do
                CURRENT_COMBINATION=$((CURRENT_COMBINATION + 1))
                
                print_separator
                echo "📝 测试组合 [$CURRENT_COMBINATION/$TOTAL_COMBINATIONS]"
                echo "分辨率：$resolution | 并发数：$concurrency"
                echo "输入Token：$input_len | 输出Token：$output_len | 请求数：$requests"
                print_separator
                
                # 运行Python测试脚本
                python "$TEST_SCRIPT" \
                    --model-name "$MODEL_NAME" \
                    --port "$PORT" \
                    --resolution "$resolution" \
                    --requests "$requests" \
                    --concurrency "$concurrency" \
                    --random-input-len "$input_len" \
                    --random-output-len "$output_len" \
                    --output "$OUTPUT_CSV"
                
                # 检查测试是否成功
                if [[ $? -eq 0 ]]; then
                    echo "✅ 测试组合完成：$resolution | $concurrency | $input_len | $output_len"
                else
                    echo "❌ 测试组合失败：$resolution | $concurrency | $input_len | $output_len"
                    # 可选：失败后继续执行其他组合
                    # exit 1
                fi
                
                # 可选：测试间隔（避免服务压力过大）
                sleep 2
            done
        done
    done
done

# 第五步：测试完成
print_separator
echo "🎉 所有测试组合执行完成！"
echo "📊 测试结果已保存至：$OUTPUT_CSV"
print_separator