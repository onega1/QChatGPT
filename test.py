import openai
from pkg.openai.modelmgr import ModelRequest, create_openai_model_request

openai.api_key = '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'
completion_api_params = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.9,  # 数值越低得到的回答越理性，取值范围[0, 1]
    "max_tokens": 512,  # 每次获取OpenAI接口响应的文字量上限, 不高于4096
    "top_p": 1,  # 生成的文本的文本与要求的符合度, 取值范围[0, 1]
    "frequency_penalty": 0.2,
    "presence_penalty": 1.0,
}
prompts = [
                {
                    'role': 'user',
                    'content': '你好'
                }
            ]
ai: ModelRequest = create_openai_model_request(completion_api_params['model'], 'user')
ai.request(
    prompts,
    **completion_api_params
)
response = ai.get_response()
print(response)    
