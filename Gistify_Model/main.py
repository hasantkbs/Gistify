from core.summarizer import summarize_long_text
from input_handlers.pdf_handler import read_pdf_text
from input_handlers.docx_handler import read_docx_text
import os
import sys
from api import database

def main_cli():
    print("Gistify Summarization CLI")
    print("------------------------")

    while True:
        print("\nChoose input type:")
        print("1. Text (enter directly)")
        print("2. PDF file")
        print("3. DOCX file")
        print("4. Exit")

        choice = input("Enter your choice (1/2/3/4): ").strip()

        text_to_summarize = ""
        file_path = ""

        if choice == '1':
            text_to_summarize = input("Enter the text you want to summarize: ")
        elif choice == '2':
            file_path = input("Enter the path to the PDF file: ").strip()
            if not os.path.exists(file_path):
                print(f"Error: File not found at {file_path}")
                continue
            try:
                text_to_summarize = read_pdf_text(file_path)
            except Exception as e:
                print(f"Error reading PDF file: {e}")
                continue
        elif choice == '3':
            file_path = input("Enter the path to the DOCX file: ").strip()
            if not os.path.exists(file_path):
                print(f"Error: File not found at {file_path}")
                continue
            try:
                text_to_summarize = read_docx_text(file_path)
            except Exception as e:
                print(f"Error reading DOCX file: {e}")
                continue
        elif choice == '4':
            print("Exiting Gistify CLI. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")
            continue

        if text_to_summarize:
            print("\nSummarizing...")
            summary = summarize_long_text(text_to_summarize)
            print("\n--- Summary ---")
            print(summary)
            print("---------------\n")
        else:
            print("No text to summarize. Please try again.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "init-db":
        print("Initializing the database...")
        database.init_db()
        print("Database initialization complete.")
    else:
        main_cli()
