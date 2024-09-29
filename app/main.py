import streamlit as st
import json

import app.upload_pipeline.split_data as split_data
import app.text_processing.speech_to_text as speech_to_text
from app.text_processing.llm_analyzer import llm_output
from app.text_processing.processor import text_analyzer, analyze_word_breaks
import app.face_detection.video_analyzer as video_analyzer
from app.voice_recognition.audio_analyser import analyze_audio
import app.text_processing.text_comparator as text_comparator
import app.face_detection.subtitles_extractor as subtitles_extractor

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
        text, words = speech_to_text.annotate(audio_video)
        subtitle = subtitles_extractor.extract_subtitles_from_frames(audio_video.frames)
        lev_similarity, cos_similarity = text_comparator.compare_subtitles_and_transcriptions(subtitle,text)
        print("lev_similarity: ",lev_similarity, " cos_similarity: ",cos_similarity)

        # text processors
        false_words, questions, tags, text_summary = llm_output(text)
        text_json = text_analyzer(text)
        st.json(text_json)
        average_break, longest_break, longest_break_start = analyze_word_breaks(words)

        # audio processors
        audio_json = analyze_audio(audio_video.audio)
        st.json(audio_json)

        st.text(text)
        # Create two tabs
        tab1, tab2, tab3 = st.tabs(["Tabela wyników", "Pytania i Tagi", "Analiza wydarzeń"])

        # Tab 1: Tabela wyników
        with tab1:
            st.subheader("Tabela wyników")
            stats = {
                "Ilośc słów": len(words),
                "Unikalne słowe": len(set(words)),
                "Średnia przerwa między słowami": average_break,
                "Najdłuższa przerwa między słowami": longest_break, 
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

        video_analyzer_output = video_analyzer.process_images(audio_video.frames,audio_video.fps)
        
        with tab3:
            st.subheader("Analiza wideo")
            
            # Filter video_analyzer_output based on criteria
            filtered_output = []
            for frame in video_analyzer_output:
                if (not frame['face']['visible'] or 
                    frame['body']['gesture'] or 
                    frame['environment']['background_people']):
                    filtered_output.append(frame)
            
            # Add longest break information
            filtered_output.append({
                "timestamp": longest_break_start,
                "event": "Najdłuższa przerwa między słowami",
                "duration": longest_break
            })
            
            # Sort the filtered output by timestamp
            filtered_output.sort(key=lambda x: x['timestamp'])
            
            # Display the filtered and formatted JSON
            st.json(json.dumps(filtered_output, indent=2, ensure_ascii=False))



if __name__ == "__main__":
    main()
