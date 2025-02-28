import openai
from openai import OpenAI
from openai.types.beta.threads.message_create_params import (
    Attachment,
    AttachmentToolFileSearch,
)
from config import OPENAI_API_KEY, pdf_path
import time
import os

# Retry logic for server-side errors
max_retries = 3
retry_delay = 10  # seconds

filename = "/Users/jonathanwang/Desktop/07_01_Bow_AsianFemaleRobot.pdf"
output_file = "extracted_content.txt"  # Output file to save extracted text
prompt = (
    "Extract all textual content from the provided PDF document, ensuring that no parts are skipped. "
    # "This includes headers, paragraphs, tables, and footnotes."
)

# Verify API key
api_key = OPENAI_API_KEY
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)
# Retry logic for file upload
for attempt in range(max_retries):
    try:
        with open(filename, "rb") as file_obj:
            file = client.files.create(file=file_obj, purpose="assistants")
        print(f"File uploaded: {file.id}")
        break  # Exit loop on success
    except openai.error.OpenAIError as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
        else:
            raise Exception("Maximum retry attempts reached for file upload.") from e
try:
    # Create PDF assistant
    pdf_assistant = client.beta.assistants.create(
        model="gpt-4o",
        description="An assistant to extract the contents of PDF files.",
        tools=[{"type": "file_search"}],
        name="PDF assistant",
    )
    print(f"PDF Assistant created: {pdf_assistant.id}")
except Exception as e:
    raise Exception(f"Error creating PDF assistant: {e}")

# Create thread
try:
    thread = client.beta.threads.create()
    print(f"Thread created: {thread.id}")
except Exception as e:
    raise Exception(f"Error creating thread: {e}")



# Create assistant message with file attachment
try:
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        attachments=[
            Attachment(
                file_id=file.id, tools=[AttachmentToolFileSearch(type="file_search")]
            )
        ],
        content=prompt,
    )
    print("Message sent to assistant.")
except Exception as e:
    raise Exception(f"Error sending message to assistant: {e}")

# Run thread
try:
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id=pdf_assistant.id, timeout=1000
    )
    if run.status != "completed":
        raise Exception(f"Run failed with status: {run.status}")
    print("Run completed successfully.")
except Exception as e:
    raise Exception(f"Error running thread: {e}")

# Retrieve and process messages
try:
    messages_cursor = client.beta.threads.messages.list(thread_id=thread.id)
    messages = [message for message in messages_cursor]

    if messages:
        # Extract content
        res_txt = messages[0].content[0].text.value if "content" in messages[0] else None
        if res_txt:
            # Save to file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(res_txt)
            print(f"Extracted content saved to {output_file}")

            # Format content for re-use in GPT prompts
            formatted_content = (
                f"EXTRACTED CONTENT:\n\n{res_txt}\n\nYou can now use this text as part "
                f"of a conversation or further analysis."
            )
            print(formatted_content)
        else:
            print("No content extracted from the messages.")
    else:
        print("No messages retrieved from the thread.")
except Exception as e:
    raise Exception(f"Error processing messages: {e}")
