import requests
import json
import base64
import argparse

def image_to_base64(image_path):
    """将本地图片转为 Base64 Data URI"""
    with open(image_path, "rb") as f:
        base64_str = base64.b64encode(f.read()).decode("utf-8")
    img_format = image_path.split(".")[-1].lower()
    return f"data:image/{img_format};base64,{base64_str}"

def test_qwen3_vl(port, model_name, prompt, image_path=None):
    """
    测试 Qwen3-VL-Thinking 服务
    :param port: 服务端口（9993）
    :param model_name: 模型名称（Qwen3-VL-Thinking）
    :param prompt: 提问文本
    :param image_path: 本地图片路径（可选）
    """
    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    # 构建消息体
    content = [{"type": "text", "text": prompt}]
    if image_path:
        # 添加图片 Base64 信息
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_base64(image_path)}
        })

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.7,  # 随机性
        "max_tokens": 2048,   # 最大生成长度
        "stream": False       # 非流式输出
    }

    # 发送请求
    try:
        print(f"=== 测试 {model_name}（端口 {port}）===")
        print(f"提问：{prompt}")
        if image_path:
            print(f"图片：{image_path}")
        
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=1200  # 图片推理耗时较长，延长超时
        )
        response.raise_for_status()  # 捕获 HTTP 错误
        result = response.json()

        # 解析并打印结果
        if "choices" in result:
            answer = result["choices"][0]["message"]["content"]
            print("\n【模型回答】：\n", answer)
        else:
            print("\n【响应结果】：\n", json.dumps(result, ensure_ascii=False, indent=2))

    except requests.exceptions.ConnectionError:
        print(f"错误：无法连接到 {url}，请检查服务是否启动、端口是否正确")
    except requests.exceptions.Timeout:
        print("错误：请求超时，模型推理时间过长")
    except Exception as e:
        print(f"错误：{e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"错误详情：{e.response.text}")

if __name__ == "__main__":
    # 命令行参数配置
    parser = argparse.ArgumentParser(description="测试 Qwen3-VL-Thinking 服务")
    parser.add_argument("--prompt", default="详细描述这张图片的内容", help="提问文本")
    parser.add_argument("--image", default=None, help="本地图片路径（如 test.jpg）")
    parser.add_argument("--port", default=9993, type=int, help="服务端口")
    parser.add_argument("--model", default="Qwen3-VL-Thinking", help="模型名称")
    args = parser.parse_args()

    # 执行测试
    test_qwen3_vl(
        port=args.port,
        model_name=args.model,
        prompt=args.prompt,
        image_path=args.image
    )
