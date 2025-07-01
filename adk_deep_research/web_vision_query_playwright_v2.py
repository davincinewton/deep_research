from io import BytesIO
import time
from time import sleep
from PIL import Image
from playwright.sync_api import sync_playwright
from openai import OpenAI
import os
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
# Your task is to analize the screenshots of a webpage, find the answer to the user prompt, only output in following format:
```json
{
    'isfinded': 'yes',
    'answer': 'answer to the user prompt',
    'description': 'descriptions...'
}
```
# Output format explanation:
- 'isfinded': You need to directly put 'yes' or 'no' to this field after you analysing the screenshot
- 'answer': Put answer here if you find answer. Otherwise leave this field ''
- 'description': description  of the screenshots, provide reasons if you find answer or not
# Don't hallucinate, output strictly based on information on the screenshots.
# The question you need to answer is:

"""

# revise_prompt="""
# 重新检查截屏图片中绘画里面的水果的名字和它们在画中的位置，检查之前对话中生成的结果，按要求重新输出结果
# """

revise_prompt="""
Revise the names of all the fruits and their postions on the painting.
Keywords: 12 o'clock position, clockwise.
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
model_id="koboldcpp/Qwen2.5-VL-32B-Instruct-Q8_0"#"koboldcpp/Qwen2.5-VL-72B-Instruct.i1-Q4_K_M" #"koboldcpp/Qwen2.5-VL-32B-Instruct-Q6_K"
model_id_a="qwen2.5-vl-32b-instruct" # #"gemma-3-27b-it@q8_0" "gemma-3-27b-it-qat" #
model_id_b="Qwen/Qwen2.5-VL-7B-Instruct"
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
    base_url="http://192.168.1.143:8000/v1",
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
# def inference_with_api(image_byteio, prompt, sys_prompt="You are a helpful assistant.", model_id="27b", min_pixels=512*28*28, max_pixels=2048*28*28):

def inference_with_api(image_byteio, prompt, tmessages:list, model_id, client):
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
    tmessages.append(event)

    completion = client.chat.completions.create(
        model = model_id,
        messages = tmessages,
        seed=78967,
        temperature=0,
        timeout=6000.0,
    )
    event = {
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
        }
    tmessages[-1] = event
    tmessages.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }
    )
    return completion.choices[0].message.content, tmessages

# threshold for major images
Min_Width = 300 
Min_Height = 300 
Max_Dimension=900

def web_vision_query(query:str, url, num_scroll=3)-> str:
    vresult = "None"
    with sync_playwright() as p:
        # Launch browser in headless mode
        browser = p.chromium.launch(headless=False)
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
            page.goto(url,timeout=0) 
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
                        if width >= Min_Width or height >= Min_Height:
                            major_images.append(img)
                except Exception as e:
                    print(f"Error checking image size: {str(e)}")
            print(f"Found {len(major_images)} major images (width >= {Min_Width}px, height >= {Min_Height}px)")


            # time.sleep(1)
            # Initialize list to store screenshot parts
            # screenshots = []
            scroll_position = 0
            # Initiate messages
            messages = ast.literal_eval(Ini_Messages)
            # Scroll and capture screenshots
            current_prompt = analyser_prompt+query
            # cnt_scroll = 0
            for index, img in enumerate(major_images):
                # page.evaluate(f"window.scrollTo(0, {scroll_position});")
                # page.wait_for_timeout(1000)  # Allow scroll to settle
                # scroll_position += viewport_height
                # Take screenshot
                # screenshot = None
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
                        image_format = pil_img.format or "Unknown"
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
                        resized_img.save(resized_bytes, format=image_format)
                        resized_bytes.seek(0)  # Reset position for reuse                    

                except Exception as e:
                    print(f"Error reading image metadata for image {index + 1}: {str(e)}")
                    real_width, real_height = "Unknown", "Unknown"
                    image_format = "Unknown"        
                    continue        
                # img = Image.open(screenshot)
                # img.show()
                # response, messages = inference_with_api(screenshot, current_prompt, messages, model_id=model_id_a, client=client_a)
                response, messages = inference_with_api(resized_bytes, current_prompt, messages, model_id=model_id, client=client)
                # response, messages = inference_with_api(screenshot, current_prompt, messages, model_id=model_id_g, client=client_r)
                # current_prompt = "webpage scoll one page down, try to find answer"
                # print(response)
                st_pos = response.find('```json')
                if st_pos != -1:
                    response = response[st_pos:]
                # else:
                #     print("Wrong resturn format: ", response)
                #     break
                response = repair_json(response)
                # print(response)
                try:
                    response = json.loads(response)
                    ql = response.get('isfinded', 'no')
                    if 'yes' == ql.lower().strip():
                        # response, messages = inference_with_api(screenshot, revise_prompt, messages, model_id=model_id, client=client)
                        # print(response)
                        # response, messages = revise_with_api(messages)
                        # print(response)
                        vresult = response
                        break
                    print('No answer found, scoll down...')
                    
                    if (index >= (num_scroll-1)):
                        print(f'scroll down {num_scroll} times, quit!')
                        break
                except json.JSONDecodeError:
                    print("Invalid JSON string")
                    break
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
    url = "https://www.daytonartinstitute.org/exhibits/janet-fish/"  # Replace with your target URL
    print(web_vision_query("Output the names of all the fruits on the painting in the screenshot in the order based on their arrangement in the painting clockwisely starting after the 12 o'clock position. Use the plural form of each fruit", url))