def load_model():
    # Direct download link for large files
    file_id = '1OmWDx2Vju3fq0RBwhZEZcA1zFZlHAWDX'
    url = f'https://drive.google.com/uc?export=download&id={file_id}&confirm=t'
    output = 'walmart_model.pkl'
    
    if not os.path.exists(output):
        with st.spinner('Downloading AI Model... Please wait.'):
            try:
                # Using a session to handle the "large file" warning from Google
                session = requests.Session()
                response = session.get(url, stream=True)
                with open(output, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)
            except Exception as e:
                st.error(f"Download failed: {e}")
    
    # Load the model
    try:
        return joblib.load(output)
    except Exception:
        # If loading fails, it means the file is bad. Delete it to retry.
        if os.path.exists(output):
            os.remove(output)
        st.error("Model loading failed. Re-trying download... Please refresh the page.")
        st.stop()
