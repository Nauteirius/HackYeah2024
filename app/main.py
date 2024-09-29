import streamlit as st

import app.upload_pipeline.split_data as split_data
import app.text_processing.speech_to_text as speech_to_text
from app.text_processing.llm_analyzer import llm_output
from app.text_processing.processor import text_analyzer


def main():
    uploaded_file = st.file_uploader("Wybierz plik formatu mp4", type=["mp4"])

    if uploaded_file is not None:
        # metadata
        file_details = {
            "Filename": uploaded_file.name,
            "File type": uploaded_file.type,
            "File size": f"{uploaded_file.size} bytes",
        }

        audio_video = split_data.split(uploaded_file)
        st.success("Zbieranie danych z video ✅")

        text, words = speech_to_text.annotate(audio_video)
        st.success("Zbieranie danych tekstowych ✅")

        # text processors
        st.text("Analiza tekstu ✅")
        false_words, questions, tags = llm_output(text)
        st.success("Analiza kontekstu ✅")
        text_analyzer(text)


        # Create two tabs
        tab1, tab2 = st.tabs(["Tabela wyników", "Pytania i Tagi"])

        # Tab 1: Tabela wyników
        with tab1:
            st.subheader("Tabela wyników")
            stats = {
                "Total Words": len(words),
                "Unique Words": len(set(words)),
                "False Words": len(false_words),
                
            }
            st.table(stats)

        # Tab 2: Questions and Tags
        with tab2:
            st.subheader("Pytania")
            for i, question in enumerate(questions, 1):
                st.write(f"{i}. {question}")

            st.subheader("Tagi")
            for i, tag in enumerate(tags, 1):
                st.write(f"{i}. {tag}")

        print("Full Text:\n", text)

        print("Individual Words:")
        for word in words:
            print(word)


if __name__ == "__main__":
    main()
