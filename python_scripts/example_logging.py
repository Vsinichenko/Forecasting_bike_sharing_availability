import logging
import sys
from datetime import datetime

start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_fullpath = f"../example_logging_output_{start_time}.log"


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("output.log"),
        logging.StreamHandler(sys.stdout)  # Optional: still print to console
    ]
)

# Redirect stdout and stderr to logging
class LoggerWriter:
    def __init__(self, level):
        self.level = level
        self.buffer = ""

    def write(self, message):
        if message.strip():  # Avoid empty messages
            self.level(message.strip())

    def flush(self):
        pass  # No need to flush manually

sys.stdout = LoggerWriter(logging.info)
sys.stderr = LoggerWriter(logging.error)  # Capture warnings and errors

# Example usage
print("This is a print statement")  # Captured as INFO
logging.info("This is a logging.info message")
logging.warning("This is a logging.warning message")  # WARNING level
raise ValueError("This is an exception")  # Captured as ERROR
