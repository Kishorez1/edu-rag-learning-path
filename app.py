import streamlit as st
from rag_pipeline import process_query

  # Streamlit app
st.title("Educational Content RAG with Learning Path")
st.write("Enter a query to generate a personalized Python learning path.")

  # Input query
query = st.text_input("Query (e.g., Learn Python basics):", "Learn Python functions")

  # Process query when button is clicked
if st.button("Generate Learning Path"):
      with st.spinner("Processing..."):
          try:
              result = process_query(query)
          except Exception as e:
              st.error(f"Error processing query: {e}")
              st.stop()

          # Display learning path summary
          st.subheader("Learning Path Summary")
          st.write(result['llm_response'])

          # Display learning path details
          st.subheader("Learning Path Details")
          for item in result['path_details']:
              st.write(f"**Step: {item['competency'].replace('_', ' ').title()} ({item['level']}, {item['status']})**")
              st.write(f"Summary: {item['summary']}")
              st.write(f"Content Preview: {item['preview']}")
              st.write("---")

          # Display user progress
          st.subheader("User Progress")
          for entry in result['progress']:
              st.write(f"**Query**: {entry['query']} at {entry['timestamp']}, Keywords: {entry.get('keywords', [])}")
              for meta in entry['learning_path']:
                  st.write(f"- {meta['competency'].replace('_', ' ').title()} ({meta['level']})")

          # Display recommendations
          st.subheader("Recommended Topics")
          st.write(", ".join([comp.replace('_', ' ').title() for comp in result['recommendations']]))

          # Display raw retrieval results
          st.subheader("Raw Retrieval Results")
          for item in result['raw_results']:
              st.write(f"**ID**: {item['id']}")
              st.write(f"Content: {item['content']}")
              st.write(f"Metadata: {item['metadata']}")