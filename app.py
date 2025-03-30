import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load dataset with caching
@st.cache_data
def load_data():
    data = pd.read_csv("amazon.csv")
    data = data[['product_name', 'category', 'rating', 'review_title']].drop_duplicates().reset_index(drop=True)

    # Clean the 'rating' column
    data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
    data = data.dropna(subset=['rating'])  # Remove rows with invalid ratings
    data['rating'] = data['rating'].astype(float)

    # Create the 'all_details' column for vectorization
    data['all_details'] = data['product_name'] + " " + data['category'] + " " + data['rating'].astype(str) + " " + data['review_title']
    return data

# Initialize Session State for Navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"  # Default page is Home

# Dynamic Page Navigation
if st.session_state.page == "Home":
    st.title("Welcome to the Product Recommendation System!")

    # Add a banner image
    st.image(
    "https://picsum.photos/800/400",  # Use a working placeholder image URL
    caption="Enhance your decision-making with smarter recommendations!",
    use_container_width=True
)


    # Description Section
    st.write("""
        Welcome to our app! Whether you're looking for smarter shopping suggestions,
        uploading your own datasets, or analyzing reviews, this app provides all the
        tools you need for the best insights.
    """)

    # Navigation Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Go to Product Recommendation"):
            st.session_state.page = "Product Recommendation"
    with col2:
        if st.button("Go to Upload Dataset"):
            st.session_state.page = "Upload Dataset"

# Product Recommendation Page
elif st.session_state.page == "Product Recommendation":
    st.title("Product Recommendation System")

    # Load data
    data = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")
    category_filter = st.sidebar.selectbox("Filter by Category", options=data['category'].unique())
    min_rating = st.sidebar.slider("Minimum Rating", min_value=0, max_value=5, value=3)

    # User input
    product_name = st.text_input("Enter Product Name", "")

    # Apply filters
    filtered_data = data[
        (data['category'] == category_filter) &
        (data['rating'] >= min_rating)
    ]
    filtered_data = filtered_data.reset_index(drop=True)

    if not filtered_data.empty and product_name:
        # Vectorization and FAISS indexing
        course_corpus = filtered_data['all_details']
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=5)
        X = vectorizer.fit_transform(course_corpus)
        X = np.float32(X.toarray())

        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)

        # Search for recommendations
        search_text_vector = vectorizer.transform([product_name])
        search_text_vector_array = np.float32(search_text_vector.toarray())
        distances, indices = index.search(search_text_vector_array, 5)

        # Create recommendations DataFrame
        recommendations = pd.DataFrame({
            'Product Name': [filtered_data.loc[indices[0][i], 'product_name'] for i in range(len(indices[0]))],
            'Category': [filtered_data.loc[indices[0][i], 'category'] for i in range(len(indices[0]))],
            'Rating': [filtered_data.loc[indices[0][i], 'rating'] for i in range(len(indices[0]))],
            'Review Title': [filtered_data.loc[indices[0][i], 'review_title'] for i in range(len(indices[0]))]
        })

        # Sentiment analysis
        recommendations['Sentiment'] = recommendations['Review Title'].apply(
            lambda x: 'Positive review' if TextBlob(x).sentiment.polarity > 0 else 'Negative review'
        )

        # Display results
        st.write("Recommended Products:")
        st.dataframe(recommendations)

        # Download recommendations
        st.download_button(
            label="Download Recommendations as CSV",
            data=recommendations.to_csv(index=False),
            file_name="product_recommendations.csv",
            mime="text/csv"
        )

        # Visualization: Ratings Distribution
        st.write("Ratings Distribution of Recommendations:")
        plt.figure(figsize=(6, 4))
        recommendations['Rating'] = recommendations['Rating'].astype(float)
        plt.hist(recommendations['Rating'], bins=5, color='skyblue', edgecolor='black')
        plt.title("Ratings Distribution")
        plt.xlabel("Ratings")
        plt.ylabel("Frequency")
        st.pyplot(plt)
    else:
        st.warning("No matching products found. Please try different inputs or filters.")

    # Back to Home Button
    if st.button("Back to Home"):
        st.session_state.page = "Home"

# Upload Dataset Page
elif st.session_state.page == "Upload Dataset":
    st.title("Upload Your Dataset")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset:")
        st.dataframe(user_data)
        st.success("Dataset uploaded successfully!")
    else:
        st.warning("Please upload a CSV file to proceed.")

    # Back to Home Button
    if st.button("Back to Home"):
        st.session_state.page = "Home"
