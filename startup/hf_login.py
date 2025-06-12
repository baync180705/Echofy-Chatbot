"""
This script handles the login process to Hugging Face using an access token.

It retrieves the Hugging Face access token from the `config` module and uses
the `huggingface_hub` library to authenticate. If the token is not found, the
script skips the login process.
"""

from config import HUGGING_FACE_ACCESS_TOKEN
from huggingface_hub import login

if HUGGING_FACE_ACCESS_TOKEN:
    """
    Logs in to Hugging Face using the provided access token.

    The token is retrieved from the `HUGGING_FACE_ACCESS_TOKEN` variable in the
    `config` module. If the token is valid, the login is successful.
    """
    login(token=HUGGING_FACE_ACCESS_TOKEN)
    print("Hugging Face login successful.")
else:
    """
    Skips the login process if no access token is found.

    A message is printed to indicate that the login process was skipped.
    """
    print("No HUGGING_FACE_ACCESS_TOKEN found. Skipping login.")
