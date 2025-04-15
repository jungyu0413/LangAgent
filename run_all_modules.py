from va_main_module import compute_va_change_average
from exp_main_module import analyze_expression_changes

def collect_all_module_results(video_path):
    va_result = compute_va_change_average(video_path)
    expr_result = analyze_expression_changes(video_path)

    results = {
        "video_path": video_path,
        "va": va_result,
        "expression": expr_result
        # 추후: "blink": blink_result 등 추가 가능
    }

    return results


# 예시 실행용 (테스트)
if __name__ == "__main__":
    video_path = "/home/face/Desktop/LangAgent/langchain_demo.mp4"
    results = collect_all_module_results(video_path)

    import json
    print(json.dumps(results, indent=2, ensure_ascii=False))
