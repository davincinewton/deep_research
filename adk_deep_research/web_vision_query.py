from io import BytesIO
import time
from time import sleep
from PIL import Image
from selenium import webdriver
from openai import OpenAI
import os
import base64

import random
import string
import json
from json_repair import repair_json

def generate_random_string():
    characters = string.ascii_letters + string.digits  # a-z, A-Z, 0-9
    return ''.join(random.choice(characters) for _ in range(6))

#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
sys_prompt="""
# You are a helpful assistant. 
# Your task is to analize the screenshots of a webpage, find the answer to the user prompt, then output in following format:
{
    'isfinded': 'yes',
    'answer': 'answer to the user prompt',
    'explanation': 'detailed explanation'
}
# Output format explanation:
- 'find_answer': You need to directly put 'yes' or 'no' to this field after you analysing the screenshot
- 'answer': Put answer here if you find answer. Otherwise leave this field ''
- 'explanation': Detailed explanation if you find answer or not
"""

ini_messages=[
    {
        "role": "system",
        "content": [{"type":"text","text": sys_prompt}]
    },

]

# model_id="koboldcpp/Qwen2.5-VL-7B-Instruct-Q5_K_M"
model_id="koboldcpp/Qwen25-VL-32B-Instruct-Q4_K_M "
# model_id="27b"
# base64_image = encode_image(image_path)
# client = OpenAI(
#     api_key="EMPTY",
#     base_url="http://localhost:5001/v1",
# )
client = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.1.143:5001/v1",
)
# client = OpenAI(
#     api_key="lm-studio",
#     base_url="http://192.168.1.143:1234/v1",
# )
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

def inference_with_api(image_byteio, prompt, messages:list):
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
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            },
            {"type": "text", "text": prompt},
        ],
    }
    messages.append(event)

    completion = client.chat.completions.create(
        model = model_id,
        messages = messages,
       
    )
    messages.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }
    )
    return completion.choices[0].message.content, messages

def web_vision_query(query:str, url)-> str:
    # Set up Selenium WebDriver (using Chrome)
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')  # Run in headless mode (no browser window)
    # options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument("--force-device-scale-factor=1")
    # options.add_argument("--window-size=960,1040")
    # options.add_argument("--disable-pdf-viewer")
    # options.add_argument("--window-position=0,0")
    driver = webdriver.Chrome(options=options)
    try:
        # Navigate to the URL
        driver.get(url)
        time.sleep(1)  # Allow page to load

        # Get total height of the page
        total_height = driver.execute_script("return document.body.scrollHeight")
        viewport_height = driver.execute_script("return window.innerHeight")
        viewport_width = driver.execute_script("return document.body.scrollWidth")

        # Set window size to capture full width
        driver.set_window_size(viewport_width, viewport_height)
        # time.sleep(1)
        # Initialize list to store screenshot parts
        # screenshots = []
        scroll_position = 0
        # Initiate messages
        messages = ini_messages
        # Scroll and capture screenshots
        while scroll_position < total_height:
            driver.execute_script(f"window.scrollTo(0, {scroll_position});")
            scroll_position += viewport_height
            time.sleep(1)  # Allow scroll to settle

            # Take screenshot
            screenshot = driver.get_screenshot_as_png()
            screenshot = BytesIO(screenshot)
            # img = Image.open(screenshot)
            # img.show()
            response, messages = inference_with_api(screenshot, query, messages)
            # print(response)
            response = repair_json(response)
            print(response)
            try:
                response = json.loads(response)
                ql = response.get('isfinded', 'no')
                if 'yes' == ql.lower().strip():
                    # response, messages = critic_with_api(messages)
                    # print(response)
                    # response, messages = revise_with_api(messages)
                    # print(response)
                    break
            except json.JSONDecodeError:
                print("Invalid JSON string")
                break
            # screenshot = Image.open(screenshot)
            # screenshots.append(screenshot)
            

        # Stitch screenshots together
        # stitched_image = Image.new('RGB', (viewport_width, scroll_position))
        # offset = 0
        # for screenshot in screenshots:
        #     stitched_image.paste(screenshot, (0, offset))
        #     offset += screenshot.size[1]

        # # Save the final image
        # output_path = generate_random_string()+'.png'
        # stitched_image.save(output_path, "PNG")
        # print(f"Screenshot saved as {output_path}")

    finally:
        driver.quit()

# Example usage
# https://daytonart.emuseum.com/objects/22447/embroidery-from-uzbekistan
# https://www.daytonartinstitute.org/exhibits/janet-fish/
# list all the names of the fruits in the paint in the order based on their arrangement in the painting clockwisely starting from the 12 o'clock position. Use the plural form of each fruit
if __name__ == "__main__":
    url = "https://daytonart.emuseum.com/objects/22447/embroidery-from-uzbekistan"  # Replace with your target URL
    web_vision_query("list all the names of the fruits in the paint in the order based on their arrangement in the painting clockwisely starting from the 12 o'clock position. Use the plural form of each fruit", url)