import subprocess
import pyautogui
import time
import os

# Path to your Excel file
excel_file = 'G:\\work\\smolagent\\gaia\\65afbc8a-89ca-4ad5-8d62-355bb401f61d.xlsx'  # Replace with your Excel file path
screenshot_path = "screenshot.png"

try:
    # Open file with LibreOffice
    # Adjust path to LibreOffice executable if needed
    # libreoffice_path = "F:\Program Files\LibreOffice\program\scalc.exe"  # Default command for LibreOffice; may need full path like "/usr/lib/libreoffice/program/soffice"
    # subprocess.Popen([libreoffice_path, "--calc", os.path.abspath(excel_file)])
    libreoffice_path = "F:\Program Files\LibreOffice\program\soffice.exe"  # Default command for LibreOffice; may need full path like "/usr/lib/libreoffice/program/soffice"
    subprocess.Popen([libreoffice_path, os.path.abspath(excel_file)])    
    
    # Wait for LibreOffice to open and display
    time.sleep(3)
    
    # Take screenshot
    screenshot = pyautogui.screenshot()
    
    # Save screenshot
    screenshot.save(screenshot_path)
    print(f"Screenshot saved as {screenshot_path}")
    
    # Close LibreOffice (optional, sends Ctrl+Q to close)
    pyautogui.hotkey('ctrl', 'q')
    
except Exception as e:
    print(f"An error occurred: {str(e)}")
    # Attempt to close LibreOffice in case of error
    try:
        pyautogui.hotkey('ctrl', 'q')
    except:
        pass