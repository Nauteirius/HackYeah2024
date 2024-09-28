import streamlit as st
import time
import json

with open('sample_data.json', "r") as file:
    data = json.load(file)

def main():
    uploaded_file = st.file_uploader("Choose an MP4 file", type=["mp4"])

    if uploaded_file is not None:
        # metadata
        # file_details = {
        #     "Filename": uploaded_file.name,
        #     "File type": uploaded_file.type,
        #     "File size": f"{uploaded_file.size} bytes"
        # }

        # st.json(file_details)

        if st.button("Process"):
            steps = [
                "Przetwarzanie wideo",
                "Przetwarzanie audio",
                "Analizowanie wypowiedzi"
            ]
            
            placeholders = [st.empty() for _ in steps]
            
            for i, step in enumerate(steps):
                placeholders[i].text(step)
            
            for i, step in enumerate(steps):
                time.sleep(2)
                placeholders[i].text(f"{step} ✅")

    print(uploaded_file)

if __name__ == "__main__":
    main()