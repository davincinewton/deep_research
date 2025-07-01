from playwright.sync_api import sync_playwright
import time

def scroll_to_major_images(url, min_width=200, min_height=200):
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=False)  # Set headless=True for no UI
        page = browser.new_page()
        
        # Navigate to the webpage
        page.goto(url)
        
        # Wait for page to load
        # page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)
        # Get all image elements
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
                    if width >= min_width and height >= min_height:
                        major_images.append(img)
            except Exception as e:
                print(f"Error checking image size: {str(e)}")
        
        print(f"Found {len(major_images)} major images (width >= {min_width}px, height >= {min_height}px)")
        
        for index, img in enumerate(major_images):
            try:
                # Scroll to the image
                img.scroll_into_view_if_needed()
                
                # Get image details
                src = img.get_attribute('src') or img.get_attribute('data-src') or "No src"
                alt = img.get_attribute('alt') or "No alt text"
                bounding_box = img.bounding_box()
                width = bounding_box['width'] if bounding_box else "Unknown"
                height = bounding_box['height'] if bounding_box else "Unknown"
                
                print(f"Major Image {index + 1}:")
                print(f"Source: {src}")
                print(f"Alt: {alt}")
                print(f"Size: {width}x{height}px")
                
                # Optional: Add delay to observe scrolling
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing major image {index + 1}: {str(e)}")
        
        # Close browser
        browser.close()

# Example usage
if __name__ == "__main__":
    target_url = "https://onepagelove.com/tag/big-images"  # Replace with your target URL
    scroll_to_major_images(target_url, min_width=200, min_height=200)  # Adjust size thresholds as needed