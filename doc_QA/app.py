# streamlit run app.py --server.port 6006 --server.address 127.0.0.1
import time
import streamlit as st
import pandas as pd

from qa import get_query_engine

if "query_engine" not in st.session_state.keys():
    # get_query_engine() 實作在 rag.py
    st.session_state.query_engine = get_query_engine()
    
if prompt := st.chat_input("Something for Nothing"):
    # st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    query_engine = st.session_state.query_engine
    with st.chat_message("assistant"):
        start_time = time.time()
        response = query_engine.query(prompt)
        end_time = time.time()
        st.write(response.response)

        query_time = round(end_time - start_time, 2)
        st.write(f"Took {query_time} second(s)")

        details_title = f"Found {len(response.source_nodes)} document(s)"
        with st.expander(details_title, expanded=False,):
            source_nodes = []
            for item in response.source_nodes:
                node = item.node
                score = item.score
                title = node.metadata.get('file_name', None)
                if title is None:
                    title = node.metadata.get('title', 'N/A') # if the document is a webpage, use the title
                    continue
                page_label = node.metadata.get('page_label', 'N/A')
                text = node.text
                source_nodes.append({"Title": title, "Page": page_label, "Text": text, "Score": f"{score:.2f}"})
            df = pd.DataFrame(source_nodes)
            st.table(df)