import os

from openai import OpenAI
from pathlib import Path
import argparse
from tkinter import Image
import glob

import fitz
from pytesseract import pytesseract
from typing_extensions import override

from BMRetriver import load_bmretriever_model, load_pubmed_documents, retrieve_relevant_documents
from config import OPENAI_API_KEY, pdf_path, AI_MODEL, System_message_KG_Ver3, System_message_KG_Ver2, \
    System_message_KG_Ver1
from evaluate_KG import read_and_evaluate_kg_files, evaluate_kg_completeness
from extract_KG import extract_KG

_openai_client = None
_current_api_key = None
openai_api_key = os.getenv("OPENAI_API_KEY", None)
def get_openai_client():
    """
    Returns an OpenAI client instance.
    - Reuses the existing instance if the API key has not changed.
    - Creates a new instance only if the API key has been updated.
    """
    global _openai_client, _current_api_key

    # Determine the API key
    new_api_key = os.getenv("OPENAI_API_KEY")  # Default from env variables
    if openai_api_key:  # Check if a dynamically set key is available
        new_api_key = openai_api_key

    # If API key is unchanged, return the existing client
    if _openai_client is not None and new_api_key == _current_api_key:
        return _openai_client

    # If API key changed (or first time), create a new client
    if not new_api_key:
        raise ValueError("OpenAI API key is not set. Use /set_api_key to configure it.")

    _openai_client = OpenAI(api_key=new_api_key)
    _current_api_key = new_api_key  # Store current key

    return _openai_client
def validate_openai_key(api_key: str) -> bool:
    """
    Validates the given OpenAI API key by making a small test request.
    Returns True if the key is valid, otherwise False.
    """
    try:
        test_client = OpenAI(api_key=api_key)
        test_client.models.list()  # Make a simple API request
        return True
    except Exception:
        return False


def reset_openai_client(new_api_key: str):
    """
    Resets the OpenAI client when a new API key is provided.
    """
    global _openai_client, _current_api_key

    if not validate_openai_key(new_api_key):
        raise ValueError("Invalid OpenAI API key. The client was not updated.")

    # Update stored API key
    _current_api_key = new_api_key
    os.environ["OPENAI_API_KEY"] = new_api_key  # Store in environment

    # Reset client so it reinitializes on the next call
    _openai_client = None

def upload_file(file_path, purpose="assistants"):
    with open(file_path, "rb") as file:
        client = get_openai_client()
        response = client.files.create(file=file, purpose=purpose)
    return response

# Function to process the uploaded file
def process_file(file_id):
    client = get_openai_client()
    # Example: Retrieve file content
    response = client.files.retrieve(file_id)
    return response

# Function to generate knowledge graph using OpenAI API
def get_KG_once(pdf_path, model="gpt-4o-mini"):
    # Upload the file to OpenAI
    upload_response = upload_file(pdf_path, purpose="assistants")
    file_id = upload_response.id  # Use dot notation to access the file ID

    # Define user message to generate knowledge graph
    user_message = (
        f"<Knowledge Graph Creation>: Please provide a knowledge graph after analyzing the \" document \"  in  \" attachment.\" Search the required files in Vector Stores in attachment ."
        f"Process the document attached and iterate over the knowledge graph creation in your next response based on the system message provided. "
        "Complete all iterations without asking for further input in subsequent chats and return the final knowledge graph in the format below:\n"
        "### First Iteration\n"
        "```turtle\n"
        "<KG content>\n"
        "```\n"
        "### Second Iteration\n"
        "```turtle\n"
        "<KG content>\n"
        "```\n"
        "### Third Iteration\n"
        "```turtle\n"
        "<KG content>\n"
        "```\n"
        "### Fourth Iteration\n"
        "```turtle\n"
        "<KG content>\n"
        "```\n"
        "### Final Iteration\n"
        "```turtle\n"
        "<KG content>\n"
        "```\n"
        f"Provide me response \"from document provided\" in \"attachments\" inside \"vector store\" with file_id \"{file_id}\".\n"
    )
    tools = [{"type": "file_search"}]

    client = get_openai_client()
    # Call OpenAI API
    response = client.chat.completions.create(
        model=model,  # Use model from config
        messages=[
            {"role": "system", "type": "text","content": System_message_KG_Ver3},
            {"role": "user",
             "content": user_message,
             "attachments": [
                 {"file_id": file_id, "tools": [{"type": "file_search"}]}
             ],
             },
        ],
        temperature=0.4,
        top_p=0.3,
    )

    # Return the response
    return response.choices[0].message.content

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file. Use OCR for image-based PDFs and direct text extraction for native PDFs.
    :param pdf_path: Path to the PDF file.
    :return: Extracted text.
    """
    pdf_text = ""
    pdf_document = fitz.open(pdf_path)

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]

        # Try to extract text directly
        text = page.get_text()
        if text.strip():
            pdf_text += text
        else:
            # If no text is found, use OCR on the page image
            pix = page.get_pixmap()
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(image)
            pdf_text += text

    pdf_document.close()
    return pdf_text

def getKGwithtext_ver3(extracted_text, model="gpt-4o-mini",temperature=0.5, top_p=0.9):
    """
    Extract text from the PDF file and directly feed it into the OpenAI model to generate a knowledge graph.
    :param pdf_path: Path to the PDF file.
    :param model: OpenAI model to use.
    """
    # Extract text from the PDF
    user_message = (
        f"<Knowledge Graph Creation>: Please provide a knowledge graph after analyzing the \" text Provided in \"Text:\"."
        f"Process the text and iterate over the knowledge graph creation in your next response based on the system message provided. "
        "Complete all iterations without asking for further input in subsequent chats and return the final knowledge graph in the format below:\n"
        "### First Iteration\n"
        "```turtle\n"
        "<KG content>\n"
        "```\n"
        "### Second Iteration\n"
        "```turtle\n"
        "<KG content>\n"
        "```\n"
        "### Third Iteration\n"
        "```turtle\n"
        "<KG content>\n"
        "```\n"
        "### Fourth Iteration\n"
        "```turtle\n"
        "<KG content>\n"
        "```\n"
        "### Final Iteration\n"
        "```turtle\n"
        "<KG content>\n"
        "```\n"
        f"Text:\n{extracted_text}"
    )
    client = get_openai_client()
    # Call the OpenAI API to generate a knowledge graph
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": System_message_KG_Ver3},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        top_p=top_p,
    )
    # Extract and print the response
    return response.choices[0].message.content

def getKGwithtext_ver2(extracted_text, model="gpt-4o-mini",temperature=0.5, top_p=0.9):
    """
    Extract text from the PDF file and directly feed it into the OpenAI model to generate a knowledge graph.
    :param pdf_path: Path to the PDF file.
    :param model: OpenAI model to use.
    """
    # Extract text from the PDF
    user_message = (
        f"<Knowledge Graph Creation>: Please provide a knowledge graph after analyzing the \" text Provided in \"Text:\"."
        f"Process the text and iterate over the knowledge graph creation in your next response based on the system message provided. "
        "Complete all iterations without asking for further input in subsequent chats and return the final knowledge graph in the format below:\n"
        "### First Iteration\n"
        "```turtle\n"
        "<KG content>\n"
        "```\n"
        "### Second Iteration\n"
        "```turtle\n"
        "<KG content>\n"
        "```\n"
        "### Third Iteration\n"
        "```turtle\n"
        "<KG content>\n"
        "```\n"
        "### Fourth Iteration\n"
        "```turtle\n"
        "<KG content>\n"
        "```\n"
        "### Final Iteration\n"
        "```turtle\n"
        "<KG content>\n"
        "```\n"
        f"Text:\n{extracted_text}"
    )
    client = get_openai_client()
    # Call the OpenAI API to generate a knowledge graph
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": System_message_KG_Ver2},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        top_p=top_p,
    )
    # Extract and print the response
    return response.choices[0].message.content
def getKGwithtext_ver1(extracted_text, model="gpt-4o-mini",temperature=0.5, top_p=0.9):
    """
    Extract text from the PDF file and directly feed it into the OpenAI model to generate a knowledge graph.
    :param pdf_path: Path to the PDF file.
    :param model: OpenAI model to use.
    """
    # Extract text from the PDF

    user_message = (
        f"<Knowledge Graph Creation>: Please provide a knowledge graph after analyzing the \" text Provided in \"Text:\"."
        f"Process the text and iterate over the knowledge graph creation in your next response based on the system message provided. "
        "Without asking for further input in subsequent chats, return the final knowledge graph in the format below:\n"
        "### Final Iteration\n"
        "```turtle\n"
        "<KG content>\n"
        "```\n"
        f"Text:\n{extracted_text}"
    )
    client = get_openai_client()
    # Call the OpenAI API to generate a knowledge graph
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": System_message_KG_Ver1},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        top_p=top_p,
    )
    # Extract and print the response
    return response.choices[0].message.content

# Extract the final KG content
def extract_final_kg(content):
    start = content.find("### Final Iteration")
    if start != -1:
        start = content.find("```turtle", start) + len("```turtle\n")
        end = content.find("```", start)
        if end != -1:
            return content[start:end].strip()
    return None
def get_KG_mul_times(extracted_text, output_dir="./", model="gpt-4o-mini",num_kg_integrated=5, temperature=0.5, top_p=0.9):
    os.makedirs(output_dir, exist_ok=True)
    all_kg_files = []
    for i in range(1, num_kg_integrated + 1):
        print(f"Generating KG {i}...")
        try:
            knowledge_graph = getKGwithtext_ver2(extracted_text, model, temperature=temperature, top_p=top_p)
            final_kg_content = extract_final_kg(knowledge_graph)
            if final_kg_content:
                final_output_filename = os.path.join(output_dir, f"final_kg_{i}.txt")
                with open(final_output_filename, "w") as file:
                    file.write(final_kg_content)
                all_kg_files.append(final_output_filename)
                print(f"Final knowledge graph content saved to {final_output_filename}")
            else:
                print(f"Failed to extract final KG content for KG {i}.")
        except Exception as e:
            print(f"An error occurred during KG {i} generation: {str(e)}")
    return all_kg_files

# Consolidate the combined KG
def consolidate_kg(kg_file_path, consolidated_output="./consolidated_kg.txt", model="gpt-4o-mini", temperature=0.9, top_p=0.9):
    # Read the content of the file
    with open(kg_file_path, "r") as infile:
        kg_content = infile.read()

    # Define user message for consolidating the knowledge graphs
    user_message = (
        "Please combine the following five KGs into one, consolidating and only retaining consistent relationships and entities. "
        "Ensure that information reliability is the top priority,so don't include relationship not solid or supported by majority. Directly return the consolidated result in the following format: "
        '### Final KG\n"""```turtle\n<KG content>\n```\n""".\n\n'
        f"### Input KGs\n```\n{kg_content}\n```"
    )
    client = get_openai_client()
    # Call OpenAI API
    response = client.chat.completions.create(
        model=model,  # Use the desired OpenAI model
        messages=[
            {"role": "system", "content": "You are an assistant that processes and consolidates knowledge graphs. You need to look at different knowledge graphs and extract the most valuable, consistent and solid information from knowledge graphs and form a final knowledge graph. You need to drop unreliable information given number of presense in graphs."},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        top_p=top_p,
    )

    # Extract the response
    consolidated_kg = response.choices[0].message.content

    # Save the consolidated KG to a file
    with open(consolidated_output, "w") as outfile:
        outfile.write(consolidated_kg)

    print(f"Consolidated knowledge graph saved to {consolidated_output}")

# Combine all generated KGs into one file
def combine_all_KGs(all_kg_files, combined_file="./combined_kgs.txt"):
    try:
        with open(combined_file, "w") as outfile:
            for idx, kg_file in enumerate(all_kg_files, start=1):
                with open(kg_file, "r") as infile:
                    outfile.write(f"### KG {idx}\n")
                    outfile.write("```\n")
                    outfile.write(infile.read())
                    outfile.write("\n```\n")
        print(f"All KGs combined into {combined_file}")
    except Exception as e:
        print(f"An error occurred during KG combination: {str(e)}")

def generate_and_compare_kgs(pdf_path, output_dir="./", model="gpt-4o-mini", num_kg_integrated=5, num_kg_compared=5, temperature=0.5, top_p=0.9):
    os.makedirs(output_dir, exist_ok=True)
    extracted_text = extract_text_from_pdf(pdf_path)
    for i in range(1, num_kg_compared + 1):
        print(f"Generating KG Set {i}...")
        # Generate multiple KGs
        cur_output_dir = f"{output_dir}_{i}"
        kg_files = get_KG_mul_times(extracted_text=extracted_text, output_dir=cur_output_dir, model=model, num_kg_integrated=num_kg_integrated, temperature=temperature, top_p=top_p)

        if num_kg_integrated == 1:
            # If only one KG is generated, no need to combine or consolidate
            print("Only one KG generated, directly processing the final KG.")
            final_output_filename = kg_files[0]  # First and only file in `kg_files`
            extract_KG(file_path=final_output_filename, output_kg_file=f"final_kg_{i}.pkl",
                       output_pdf_file=f"Final_KG_{i}.pdf")
        else:
            # Combine generated KGs
            combined_file = os.path.join(cur_output_dir, f"combined_kgs_{i}.txt")
            combine_all_KGs(kg_files, combined_file=combined_file)

            # Consolidate the combined KG
            consolidated_output = os.path.join(cur_output_dir, f"consolidated_kg_{i}.txt")
            consolidate_kg(combined_file, consolidated_output=consolidated_output, model=model,temperature=temperature, top_p=top_p)

            # Extract KG and save graph visualization
            extract_KG(file_path=consolidated_output, output_kg_file=f"final_kg_{i}.pkl",
                       output_pdf_file=f"Final_KG_{i}.pdf")
    file_paths = [f"final_kg_{i}.pkl" for i in range(1, num_kg_compared+1)]
    read_and_evaluate_kg_files(file_paths)
    schema_file_path = "schema.json"  # Path to your schema JSON file
    evaluate_kg_completeness(schema_file_path, file_paths)

def elaborate_consolidation_KG(
    kg_file_path,
    output_file="./comprehensive_kg.txt",
    model="gpt-4o-mini",
    temperature=0.9,
    top_p=0.9
):
    """
    Similar to consolidate_kg, but aims to produce a more *comprehensive* KG
    by integrating complementary or partial data rather than strictly discarding
    minority or uncertain info.
    """
    with open(kg_file_path, "r", encoding="utf-8") as infile:
        kg_content = infile.read()

    user_message = (
        "Combine the knowledge graphs below into a single, comprehensive KG. "
        "Retain complementary or additional info; if there's direct contradiction, "
        "discard or clarify. Return the final result in the format:\n"
        "### Comprehensive KG\n```turtle\n<KG content>\n```\n\n"
        f"### Input KGs\n```\n{kg_content}\n```"
    )
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that merges multiple knowledge graphs into a single, comprehensive graph."
            },
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
        top_p=top_p,
    )

    merged_kg = response.choices[0].message.content
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write(merged_kg)

    print(f"Comprehensive KG saved to {output_file}")
    return output_file
def generateKGFromText(
    text: str,
    output_dir="./",
    model="gpt-4o-mini",
    num_kg_integrated=5,
    temperature=0.5,
    top_p=0.9
):
    """
    Similar to generate_and_compare_kgs but for a single text input:
    - Generate num_kg_integrated KGs (via get_KG_mul_times).
    - If only 1 KG is integrated, that is the final KG.
    - Otherwise, combine and consolidate into a single final KG.
    - Return the path to that final KG file (text).

    Also produce a .pkl and .pdf using `extract_KG`.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Generate multiple KGs from this text
    kg_files = get_KG_mul_times(
        extracted_text=text,
        output_dir=output_dir,
        model=model,
        num_kg_integrated=num_kg_integrated,
        temperature=temperature,
        top_p=top_p
    )

    if not kg_files:
        print("No KGs were generated.")
        return None

    # 2) If num_kg_integrated == 1, skip consolidation
    if num_kg_integrated == 1:
        print("Only one KG generated; no consolidation needed.")
        final_kg_file = kg_files[0]

        # Extract to produce pkl/pdf
        extract_KG(
            file_path=final_kg_file,
            output_kg_file=os.path.join(output_dir, "final_kg.pkl"),
            output_pdf_file=os.path.join(output_dir, "final_kg.pdf")
        )
        return final_kg_file

    # 3) Combine & consolidate
    combined_file = os.path.join(output_dir, "combined_kgs.txt")
    consolidate_file = os.path.join(output_dir, "consolidated_kg.txt")

    combine_all_KGs(kg_files, combined_file=combined_file)
    consolidate_kg(combined_file, consolidated_output=consolidate_file,
                   model=model, temperature=temperature, top_p=top_p)

    # 4) Extract final KG to produce .pkl and .pdf
    extract_KG(
        file_path=consolidate_file,
        output_kg_file=os.path.join(output_dir, "final_kg.pkl"),
        output_pdf_file=os.path.join(output_dir, "final_kg.pdf")
    )

    return consolidate_file

def generate_KG_ver1(
    text_list,
    output_dir="./",
    model="gpt-4o-mini",
    num_kg_integrated=5,
    temperature=0.5,
    top_p=0.9
):
    """
    Takes multiple text segments, merges them into one large text,
    calls generateKGFromText(...), and returns the final KG path.
    """
    # 1) Combine all text segments
    combined_text = "\n\n".join(text_list)

    # 2) Generate a single final KG from the combined text
    final_kg_path = generateKGFromText(
        text=combined_text,
        output_dir=output_dir,
        model=model,
        num_kg_integrated=num_kg_integrated,
        temperature=temperature,
        top_p=top_p
    )
    return final_kg_path

def generate_KG_ver2(
    text_list,
    output_dir="./",
    model="gpt-4o-mini",
    num_kg_integrated=5,
    temperature=0.5,
    top_p=0.9
):
    """
    For each text in text_list, call generateKGFromText(...) to get a final KG.
    Then combine them all, and do an 'elaborate' consolidation (comprehensive merging).
    Returns the path to that final merged KG file.
    """
    os.makedirs(output_dir, exist_ok=True)
    final_paths = []

    # 1) Generate an individual final KG for each text
    for idx, text_str in enumerate(text_list, start=1):
        subdir = os.path.join(output_dir, f"kg_text_{idx}")
        kg_path = generateKGFromText(
            text=text_str,
            output_dir=subdir,
            model=model,
            num_kg_integrated=num_kg_integrated,
            temperature=temperature,
            top_p=top_p
        )
        if kg_path:
            final_paths.append(kg_path)

    if not final_paths:
        print("No KGs were generated from the list of texts.")
        return None

    # 2) Combine these final KGs
    combined_kgs_file = os.path.join(output_dir, "combined_ver2_kgs.txt")
    combine_all_KGs(final_paths, combined_file=combined_kgs_file)

    # 3) Use the 'elaborate_consolidation_KG' to merge complementary data
    comprehensive_file = os.path.join(output_dir, "comprehensive_kg_ver2.txt")
    elaborate_consolidation_KG(
        kg_file_path=combined_kgs_file,
        output_file=comprehensive_file,
        model=model,
        temperature=temperature,
        top_p=top_p
    )

    return comprehensive_file

def generate_summary_from_KG(
    text:list,
    mode: str,
    query: str,
    model="gpt-4o-mini",
    temperature=0.5,
    top_p=0.9,
    output_dir="./KG_Summary"
):
    """
    1) Based on 'mode', generate a final KG from the text (ver1 or ver2).
    2) Read that KG's contents from disk.
    3) Construct a custom prompt that includes both:
       - The original text
       - The final KG
       - The user query
       (No call to generate_summary_from_texts; we do our own prompt)
    4) Call GPT to produce a final summary as plain text.
    5) Return the summary.
    """
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # 1) Generate final KG
    # -------------------------
    if mode.lower() == "ver1":
        final_kg_path = generate_KG_ver1(
            text_list=text,
            output_dir=output_dir,
            model=model,
            num_kg_integrated=1,      # or more if you like
            temperature=temperature,
            top_p=top_p
        )
    elif mode.lower() == "ver2":
        # ver2 expects a list of texts; supply single
        final_kg_path = generate_KG_ver2(
            text_list=text,
            output_dir=output_dir,
            model=model,
            num_kg_integrated=3,
            temperature=temperature,
            top_p=top_p
        )
    else:
        raise ValueError("mode must be either 'ver1' or 'ver2'")

    if not final_kg_path:
        return "No final KG was generated."

    # -------------------------
    # 2) Read KG from disk
    # -------------------------
    with open(final_kg_path, "r", encoding="utf-8") as f:
        kg_text = f.read()

    # -------------------------
    # 3) Build a custom prompt
    # -------------------------
    user_message = (
        "You have the following text (relevant source material):\n"
        f"--- BEGIN TEXT ---\n{text}\n--- END TEXT ---\n\n"
        "And the following Knowledge Graph (KG):\n"
        f"--- BEGIN KG ---\n{kg_text}\n--- END KG ---\n\n"
        "User query:\n"
        f"{query}\n\n"
        "Please write a concise plain-text summary that addresses this query, "
        "using only the information from the text and the KG. "
        "Do not include extraneous details. Return the summary in plain text.\n"
    )

    # -------------------------
    # 4) Call GPT to produce summary
    # -------------------------
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a knowledge assistant that merges a text and a KG to answer user queries."
            },
            {
                "role": "user",
                "content": user_message
            },
        ],
        temperature=temperature,
        top_p=top_p,
    )

    # Get final summary text
    summary = response.choices[0].message.content
    return summary
def describeThroughKG(
    query: str,
    retrieved_texts,
    mode="ver1",
    ai_model="gpt-4o-mini",
    temperature=0.5,
    top_p=0.9,
    top_k=3
):
    """
    1. Retrieve `top_k` relevant documents for the `query`.
    2. Combine those doc texts into a single text (or multiple texts).
    3. Pass them + the chosen mode to generate_summary_from_KG.
    4. Return the resulting summary.

    Similar to describeThroughDirectText, but calls generate_summary_from_KG instead.
    """
     # Use generate_summary_from_KG with the chosen mode
    summary = generate_summary_from_KG(
        text=retrieved_texts,
        mode=mode,
        query=query,
        model=ai_model,
        temperature=temperature,
        top_p=top_p
    )
    output_dir = "./KG_Summary/"
    if mode.lower() == "ver1":
        pdf_path = os.path.join(output_dir, "final_kg.pdf")  # Single PDF
        return summary, [pdf_path]  # Return as list for consistency
    elif mode.lower() == "ver2":
        pdf_files = [os.path.join(output_dir, f"kg_text_{i}", "final_kg.pdf") for i in range(1, top_k + 1)]
        # Only keep files that actually exist
        selected_pdfs = [pdf for pdf in pdf_files if os.path.exists(pdf)]
        return summary, selected_pdfs

    return summary,None

def generate_summary_from_texts(
    texts,
    query,
    model="gpt-4o-mini",
    temperature=0.5,
    top_p=0.9
):
    """
       Generates a summary from a list of texts based on the given query.

       :param texts: List of strings, each representing a text segment.
       :param query: A string query describing what the summary should focus on.
       :param model: The OpenAI model to use (default: 'gpt-4o-mini').
       :param temperature: Controls the randomness of the model's output (default: 0.5).
       :param top_p: Controls the nucleus sampling (default: 0.9).
       :return: A string containing the generated summary.
       """
    # Join the texts into a single prompt section
    combined_texts = "\n\n".join(texts)
    user_message = (
        "You are given multiple text segments. You will write a concise summary "
        "that addresses the user's query using **only** the information from these texts. "
        "Do not include extraneous details. Provide your summary in plain text.\n\n"
        f"---\n"
        f"**Texts**:\n{combined_texts}\n\n"
        f"**Query**:\n{query}\n\n"
        "Now please provide your summary."
    )
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an assistant that summarizes multiple texts to answer a user query."
            },
            {
                "role": "user",
                "content": user_message
            },
        ],
        temperature=temperature,
        top_p=top_p,
    )
    summary = response.choices[0].message.content
    return summary
def describeThroughDirectText(query: str, retrieved_texts, ai_model="gpt-4o-mini", top_k=3,temperature=0.5, top_p=0.9):
    summary = generate_summary_from_texts(retrieved_texts, query, model=ai_model, temperature=temperature, top_p=top_p)
    return summary
# Argument parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate knowledge graphs from a PDF file using OpenAI API.")
    parser.add_argument("--pdf_path", type=str, default=pdf_path, help="Path to the input PDF file.")
    parser.add_argument("--output_dir", type=str, default="./KGs", help="Directory to save KGs and outputs.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use.")
    parser.add_argument("--num_kg_integrated", type=int, default=2, help="Number of KGs to generate per set.")
    parser.add_argument("--num_kg_compared", type=int, default=3, help="Number of KG comparison sets.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for OpenAI API.")
    parser.add_argument("--top_p", type=float, default=0.1, help="Top-p sampling for OpenAI API.")
    parser.add_argument("--mode", type=str, default="ver1", help="Which function to demo: 'ver1' or 'ver2'.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top documents to retrieve.")
    args = parser.parse_args()
    # try:
    #     generate_and_compare_kgs(
    #         pdf_path=args.pdf_path,
    #         output_dir=args.output_dir,
    #         model=args.model,
    #         num_kg_integrated=args.num_kg_integrated,
    #         num_kg_compared=args.num_kg_compared,
    #         temperature=args.temperature,
    #         top_p=args.top_p,
    #     )
    # except Exception as e:
    #     print(f"An error occurred: {str(e)}")
    # Sample usage for demonstration:

    retrieveModel, tokenizer = load_bmretriever_model()
    documents = load_pubmed_documents(num_docs=50)  # or however many
    query = "Give me information regarding tumor inhibiting properties"
    retrieved_texts = retrieve_relevant_documents(
        query=query,
        model=retrieveModel,
        tokenizer=tokenizer,
        documents=documents,
        top_k=args.top_k
    )
    if not retrieved_texts:
        print("No documents found.")
        exit(0)
    print(retrieved_texts)
    print("\n=== Using describeThroughKG ===")
    # 2) Summarize by generating a KG from the relevant documents
    summary_describe_kg = describeThroughKG(
        query=query,
        retrieved_texts=retrieved_texts,
        mode=args.mode,  # 'ver1' or 'ver2'
        ai_model=args.model,
        temperature=args.temperature,
        top_p=args.top_p
    )
    print("\n=== Using describeThroughDirectText ===")
    summary_describe_direct = describeThroughDirectText(
        query=query,
        retrieved_texts=retrieved_texts,
        ai_model = args.model,
        top_k=args.top_k,  # how many docs to retrieve
        temperature=args.temperature,
        top_p=args.top_p
    )


    print("\n[KG-based Summary]")
    print(summary_describe_kg)
    print(summary_describe_direct)

