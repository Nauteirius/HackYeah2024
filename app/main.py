import streamlit as st

def main():
    uploaded_file = st.file_uploader("Choose an MP4 file", type=["mp4"])

    if uploaded_file is not None:
        #metadata
        file_details = {
            "Filename": uploaded_file.name,
            "File type": uploaded_file.type,
            "File size": f"{uploaded_file.size} bytes"
        }

        st.json(file_details)

    print(uploaded_file)


if __name__ == "__main__":
    main()