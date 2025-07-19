import os
import sys

conf_dir = r'C:\Users\aivad\anaconda3\dcbs_project-NEW\conf'

try:
    os.makedirs(conf_dir, exist_ok=True)
    print(f"Directory created: {conf_dir}")
    sys.exit(0)
except Exception as e:
    print(f"Error creating directory {conf_dir}: {e}")
    sys.exit(1)