import openai
import json
from pkg.openai.modelmgr import ModelRequest, create_openai_model_request

class ChatGPT:
    def __init__(self):
        openai.api_key = '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'
        self.completion_api_params = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.9,  # 数值越低得到的回答越理性，取值范围[0, 1]
        "max_tokens": 512,  # 每次获取OpenAI接口响应的文字量上限, 不高于4096
        "top_p": 1,  # 生成的文本的文本与要求的符合度, 取值范围[0, 1]
        "frequency_penalty": 0.2,
        "presence_penalty": 1.0,
        }

        self.ai: ModelRequest = create_openai_model_request(self.completion_api_params['model'], 'user')

    def reply(self, str):
        prompts = [
                {
                    'role': 'user',
                    'content': str
                }
        ]
    
        self.ai.request(
            prompts,
            **self.completion_api_params
        )
        return self.ai.get_response()


if __name__ == '__main__':
    print('输入一段话:')
    str = input()
    response = ChatGPT().reply(str=str)
    res = json.dumps(response['choices'][0]['message']['content'][2:])
    # res = "\n\n\u4f60\u597d\uff0c\u6709\u4ec0\u4e48\u6211\u53ef\u4ee5\u5e2e\u52a9\u60a8\u7684\u5417\uff1f"
    print('GPT 回复:')
    print(json.loads(res))
            

