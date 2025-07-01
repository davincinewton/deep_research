from io import BytesIO
import time
from time import sleep
from PIL import Image
from playwright.sync_api import sync_playwright
from openai import OpenAI
import re
import base64
import requests
import random
import string
import json
from json_repair import repair_json
import ast

def generate_random_string():
    characters = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
    return ''.join(random.choice(characters) for _ in range(6))

#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
sys_prompt="""
# You are a helpful assistant. 

"""

analyser_prompt="""
# Your task is to analize an online image, first describe the image in details, then find the answer to the user prompt, only output in following format:
```json
{
    'isfound': 'yes',
    'answer': 'answer to the user prompt',
    'description': 'detailed descriptions...'
}
```
# Output format explanation:
- 'isfound': You need to directly put 'yes' or 'no' to this field after you analysing the screenshot
- 'answer': Put answer here if you find answer. Otherwise leave this field ''
- 'description': detailed description of the image
# Don't hallucinate.
# The user prompt is:


"""

critic_prompt = """You are an online image analysing critic. Review the online image, user question and answer provided in
previous chats. Provide 1-2 sentences of constructive criticism on how to improve it.
Output in following format:

**Critics**
Critic 1
Critic 2
...

"""

reviser_prompt="""
# You are an online image analysing reviser, Review the online image, user question and answer provided in
previous chats. And review the critics provided in previous chat, then output the revised answer in following format:
```json
{
    'isfound': 'yes',
    'answer': 'answer to the user prompt',
    'description': 'descriptions...'
}
```
"""

Ini_Messages="""
[
    {
        "role": "system",
        "content": [{"type":"text","text": "You are a helpful assistant. "}]
    },

]
"""

APIKEY_R = "AIzaSyCQXjSGS90jbfU8sT8Q5nEEZ2Ec5aj2xgc"
APIKEY_D = "AIzaSyCTYphcXmQ-PUMxc7-10_W7YQO1Eiw6gG4"

model_id_c="koboldcpp/Qwen2.5-VL-7B-Instruct-Q5_K_M"
model_id="koboldcpp/Qwen2.5-VL-72B-Instruct.i1-Q4_K_M" #"koboldcpp/Qwen2.5-VL-32B-Instruct-Q6_K" "koboldcpp/Qwen2.5-VL-32B-Instruct-Q8_0"#
model_id_a="14b" #"qwen2.5-vl-32b-instruct" # #"gemma-3-27b-it@q8_0" "gemma-3-27b-it-qat" #"gemma-3-27b-it" #
model_id_b="Qwen2.5-VL-32B-Instruct-Q8_0"
model_id_g="gemini-2.0-flash"
# model_id="27b"
# base64_image = encode_image(image_path)
client_c = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:5001/v1",
)
client = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.1.143:5001/v1",
)
client_a = OpenAI(
    api_key="lm-studio",
    base_url="http://192.168.1.157:1234/v1",
)
client_b = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.1.121:8080/v1",
)
client_r = OpenAI(
    api_key=APIKEY_R,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
client_d = OpenAI(
    api_key=APIKEY_R,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
# client = OpenAI(
#     api_key="ollama",
#     base_url="http://192.168.1.121:11434/v1",
# )
# client = OpenAI(
#     api_key="EMPTY",
#     base_url="http://192.168.1.143:8000/v1",
# )
# model_id="koboldcpp/Qwen2.5-VL-7B-Instruct-Q5_K_M"
# model_id="Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
# Qwen/Qwen2.5-VL-7B-Instruct-AWQ  mistral-small3.1:latest  gemma3:27b gemma-3-27b-it@q8_0  koboldcpp/Qwen2.5-VL-7B-Instruct-Q5_K_M
# @title inference function with API

def inference_with_api(image_byteio, prompt, tmessages:list, model_id, client):
    if image_byteio is not None:
        min_pixels=512*28*28
        max_pixels=2048*28*28
        base64_image = base64.b64encode(image_byteio.getvalue()).decode("utf-8")
        event =     {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    # Pass in BASE64 image data. Note that the image format (i.e., image/{format}) must match the Content Type in the list of supported images. "f" is the method for string formatting.
                    # PNG image:  f"data:image/png;base64,{base64_image}"
                    # JPEG image: f"data:image/jpeg;base64,{base64_image}"
                    # WEBP image: f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    else:
        event =     {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }        
    tmessages.append(event)

    completion = client.chat.completions.create(
        model = model_id,
        messages = tmessages,
        # seed=78967,
        temperature=0,
        timeout=6000.0,
    )
    tmessages.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }
    )
    return completion.choices[0].message.content, tmessages

def inference_with_api_dataurl(image_data:str, prompt, tmessages:list, model_id, client):
    if image_data is not None:
        min_pixels=512*28*28
        max_pixels=2048*28*28
        event =     {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    # Pass in BASE64 image data. Note that the image format (i.e., image/{format}) must match the Content Type in the list of supported images. "f" is the method for string formatting.
                    # PNG image:  f"data:image/png;base64,{base64_image}"
                    # JPEG image: f"data:image/jpeg;base64,{base64_image}"
                    # WEBP image: f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": image_data},
                },
                {"type": "text", "text": prompt},
            ],
        }
    else:
        event =     {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }        
    tmessages.append(event)

    completion = client.chat.completions.create(
        model = model_id,
        messages = tmessages,
        # seed=78967,
        temperature=0,
        timeout=6000.0,
    )
    tmessages.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }
    )
    return completion.choices[0].message.content, tmessages

def run_analyser(text_input: str, resized_bytes, modelid=model_id_b, clientapi=client_b, is_dataurl=False):
    messages = ast.literal_eval(Ini_Messages)
    current_prompt = analyser_prompt+text_input
    # print(messages)
    # create
    if is_dataurl:
        response, messages = inference_with_api_dataurl(resized_bytes, current_prompt, messages, model_id=modelid, client=clientapi)
    else:
        response, messages = inference_with_api(resized_bytes, current_prompt, messages, model_id=modelid, client=clientapi)
    print(response)
    # print(messages)
    # # critic
    # tprompt = critic_prompt
    # response, messages = inference_with_api(None, tprompt, messages, model_id=modelid, client=clientapi)
    # print(response)
    # # # print(messages)
    # # # revise
    # tprompt = reviser_prompt
    # response, messages = inference_with_api(None, tprompt, messages, model_id=modelid, client=clientapi)
    # print(response)   
    # print(messages)
    tmp = response.find("</think>")
    # print(tmp)
    if tmp != -1:
        response = response[tmp+8:]
        # print(response)
    response = repair_json(response)    
    return response

def run_base64_analyser(text_input: str, resized_bytes, modelid=model_id_b, clientapi=client_b):
    messages = ast.literal_eval(Ini_Messages)
    current_prompt = text_input
    # print(messages)
    # create
    response, messages = inference_with_api_dataurl(resized_bytes, current_prompt, messages, model_id=modelid, client=clientapi)
    print(response)
    # print(messages)
    # # critic
    # tprompt = critic_prompt
    # response, messages = inference_with_api(None, tprompt, messages, model_id=modelid, client=clientapi)
    # print(response)
    # # # print(messages)
    # # # revise
    # tprompt = reviser_prompt
    # response, messages = inference_with_api(None, tprompt, messages, model_id=modelid, client=clientapi)
    # print(response)   
    # print(messages)
    tmp = response.find("</think>")
    # print(tmp)
    if tmp != -1:
        response = response[tmp+8:]
        # print(response)
    # response = repair_json(response)    
    return response

# threshold for major images
Min_Width = 250 
Min_Height = 250 
Max_Dimension=900

def is_data_url(url_string)->str:
    # Find the start of the data URL part
    data_url_start_index = url_string.find("data:image/")
    if data_url_start_index == -1:
        print("fetching image: ", url_string)
        return "NO"
    else:
        print('data image in url detected: ', url_string[:data_url_start_index+50])
        dimg = url_string[data_url_start_index:]
        print(dimg[:100])
        match = re.match(r'data:(?P<mime_type>image/[a-zA-Z0-9.+]+);base64,(?P<data>.+)', dimg)

        if not match:
            print("Error: Could not parse the data URL format.")
            return "BAD"

        mime_type = match.group('mime_type')
        if mime_type.lower() not in ['image/png','image/jpeg','image/webp']:
            print('wrong image type: ', mime_type)
            return "BAD"
        # base64_data = match.group('data')        
        # print(base64_data[:50])
        return dimg

def web_vision_query(query:str, url, num_scroll=3)-> str:
    vresult = "None"
    with sync_playwright() as p:
        # Launch browser in headless mode
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()

        # Set viewport to a reasonable width, adjust as needed
        page = context.new_page()
        # viewport_width = 1200  # Fixed width, can be dynamic if needed
        viewport_width = 1200
        viewport_height = 900
        page.set_viewport_size({"width": viewport_width, "height": viewport_height})
        try:
            # grab text information first
            # TBD.
            # Navigate to the URL
            page.goto(url,timeout=30000) 
            page.wait_for_timeout(1000)
            # page.wait_for_load_state("networkidle")  # Wait for page to fully load

            # Get page dimensions
            total_height = page.evaluate("document.body.scrollHeight")
            # get major images
            images = page.query_selector_all('img, [role="img"], picture, figure')
            
            major_images = []
            for img in images:
                try:
                    # Get image dimensions
                    bounding_box = img.bounding_box()
                    if bounding_box:
                        width = bounding_box['width']
                        height = bounding_box['height']
                        # Check if image meets size threshold
                        if width >= Min_Width and height >= Min_Height:
                            major_images.append(img)
                except Exception as e:
                    print(f"Error checking image size: {str(e)}")
            print(f"Found {len(major_images)} major images (width >= {Min_Width}px, height >= {Min_Height}px)")


            # time.sleep(1)
            # Initialize list to store screenshot parts
            # screenshots = []
            # cnt_scroll = 0
            for index, img in enumerate(major_images):
                # page.evaluate(f"window.scrollTo(0, {scroll_position});")
                # page.wait_for_timeout(1000)  # Allow scroll to settle
                # scroll_position += viewport_height
                # Take screenshot
                # screenshot = None
                if index >= num_scroll:
                    break
                response = None
                img.scroll_into_view_if_needed()
                # Get image source
                src = img.get_attribute('src') or img.get_attribute('data-src')
                if not src:
                    print(f"Major Image {index + 1}: No valid source found")
                    continue
                
                # Ensure src is a full URL
                if not src.startswith(('http://', 'https://')):
                    src = page.url.rstrip('/') + '/' + src.lstrip('/')
                # print('fetching image: ', src)
                img_data = is_data_url(src)
                if img_data == "BAD":
                    continue
                if img_data == "NO":
                    # Fetch image content
                    try:
                        response = requests.get(src, timeout=10)
                        response.raise_for_status()  # Raise exception for bad status codes
                        # Convert image content to BytesIO
                        image_bytes = BytesIO(response.content)
                    
                        # Get real size and format using PIL
                        with Image.open(image_bytes) as pil_img:
                            real_width, real_height = pil_img.size
                            print('image size: ', real_width, real_height)
                            # image_format = pil_img.format or "Unknown"
                            # Reset BytesIO position to start for reuse
                            image_bytes.seek(0)
                            # Calculate new dimensions to maintain aspect ratio
                            aspect_ratio = real_width / real_height
                            if real_width > real_height:
                                new_width = min(real_width, Max_Dimension)
                                new_height = int(new_width / aspect_ratio)
                            else:
                                new_height = min(real_height, Max_Dimension)
                                new_width = int(new_height * aspect_ratio)
                            
                            # Resize image
                            resized_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                            
                            # Save resized image to a new BytesIO object
                            resized_bytes = BytesIO()
                            resized_img.save(resized_bytes, format="JPEG")
                            resized_bytes.seek(0)  # Reset position for reuse                    

                    except Exception as e:
                        print(f"Error reading image metadata for image {index + 1}: {str(e)}")
                        real_width, real_height = "Unknown", "Unknown"
                        image_format = "Unknown"        
                        continue        
                # img = Image.open(screenshot)
                # img.show()
                # response, messages = inference_with_api(screenshot, current_prompt, messages, model_id=model_id_a, client=client_a)
                # response, messages = inference_with_api(resized_bytes, current_prompt, messages, model_id=model_id, client=client)
                # response, messages = inference_with_api(screenshot, current_prompt, messages, model_id=model_id_g, client=client_r)
                # current_prompt = "webpage scoll one page down, try to find answer"
                if img_data == "NO":
                    iresponse = run_analyser(query,resized_bytes,is_dataurl=False)
                else:
                    iresponse = run_analyser(query,img_data,is_dataurl=True)
                # print(response)
                st_pos = iresponse.find('```json')
                if st_pos != -1:
                    iresponse = iresponse[st_pos:]
                # else:
                #     print("Wrong resturn format: ", response)
                #     break
                iresponse = repair_json(iresponse)
                # print(response)
                try:
                    iresponse = json.loads(iresponse)
                    ql = iresponse.get('isfound', 'no')
                    if 'yes' == ql.lower().strip():
                        # response, messages = inference_with_api(screenshot, revise_prompt, messages, model_id=model_id, client=client)
                        # print(response)
                        # response, messages = revise_with_api(messages)
                        # print(response)
                        vresult = iresponse
                        break
                    print('No answer found, scoll down...')
                    

                except json.JSONDecodeError:
                    print("Invalid JSON string")
                    break
                # if (index >= (num_scroll-1)):
                #     print(f'scroll down {num_scroll} times, quit!')
                #     break
        except Exception as e:
            print(e)
        finally:
            browser.close()
    return vresult
# https://www.yeeyi.com/index.php
# Example usage
# https://daytonart.emuseum.com/objects/22447/embroidery-from-uzbekistan  https://www.daytonartinstitute.org/art/special-exhibitions/
# https://www.daytonartinstitute.org/exhibits/janet-fish/
# list all the names of the fruits in the paint in the order based on their arrangement in the painting clockwisely starting from the 12 o'clock position. Use the plural form of each fruit
# Output the names of all the fruits in the painting on the screenshot in the order based on their arrangement in the painting clockwisely starting from the 12 o'clock position. Use the plural form of each fruit
# 从截屏图片中的绘画中提取所有水果的名字的英文复数，输出顺序按画中12点位置开始顺时针排列
if __name__ == "__main__":
    # url = "https://www.daytonartinstitute.org/exhibits/janet-fish/"  # Replace with your target URL
    # print(web_vision_query("Output the names of all the fruits on the painting in the screenshot in the order based on their arrangement in the painting clockwisely starting after the 12 o'clock position. Use the plural form of each fruit", url))

    # print(web_vision_query("Which specific fruits are depicted in the 2008 painting 'Embroidery from Uzbekistan' and do historical records confirm their presence?", url))
    # print(web_vision_query("describe this image", url))


    test = "https://www.reddit.com/r/Python/comments/9n03mt/i_wrote_a_script_that_finds_the_shortest_path/data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAVAAAAFQCAYAAADp6CbZAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAABUKADAAQAAAABAAABUAAAAAAV7ikLAABAAElEQVR4Aey9CZAlyXnfl++9vo85d2ewBxaLYwEYFwEQICCeIiGQBA1a4mGatEMKBiVTohWyGJapCDpsGQ4j6LB1WLZkWgqRYohHiBJFW7RIOChSAYIiCIGACUAgiWsP7GLP2Z2dnZm+u98r/3//L7961T2z93C3p7eyOyu//K7Mypf5f5lZ9apK6UPfAn0L9C3Qt8BzaoHBc7LqjfoWeAFboGmaWRV3ereUVzSlvFzpijru4lwpq5NSlsRbGiovnUb8if7Gw2FZU35dcVOyy0ovKV5QvE/"

    print(is_data_url(test))