# **Product Recommendation System**

Welcome to the **Product Recommendation System**! This application leverages powerful machine learning techniques to provide tailored product suggestions based on user inputs and dataset analysis. Whether you're exploring smarter shopping options or analyzing user reviews, this app has you covered!

## **Features**
- 🛍️ **Intelligent Product Recommendations**
   - Provides personalized product suggestions using FAISS for efficient vector similarity search.
   - Users can input product names and explore the closest matches.
  
- 🗂️ **Dataset Upload**
   - Supports uploading custom datasets for unique analysis and recommendations.
   - Easily accessible via a user-friendly file uploader.

- 📊 **Data Exploration and Visualization**
   - Filter recommendations based on category and rating.
   - Visualize rating distributions to better understand product performance.

- ✨ **Sentiment Analysis**
   - Analyzes user reviews to determine overall sentiment (positive/negative).
   - Helps users make informed decisions based on customer feedback.

## **How It Works**
1. **Dataset:** The app analyzes product details (e.g., name, category, ratings, and reviews) from a CSV dataset (`amazon.csv`).
2. **Recommendation Engine:**  
   - Utilizes TF-IDF for text vectorization and FAISS for similarity search to generate top product matches.
3. **Interactive Filters:** Users can refine results using dynamic filters for categories and ratings.
4. **Sentiment Insights:** Reviews are processed using TextBlob for polarity scoring to identify customer sentiment.

## **Technologies Used**
- 🐍 **Python**: The backbone of the application logic.
- 🖥️ **Streamlit**: Ensures a smooth and interactive user experience.
- 📚 **Pandas & Numpy**: For efficient data manipulation and numerical computations.
- 🧠 **Scikit-learn & FAISS**: Implements advanced vector search and machine learning techniques.
- 🗒️ **TextBlob**: Performs sentiment analysis on reviews.
- 📈 **Matplotlib**: Visualizes data insights for better understanding.

## **Setup Instructions**
1. Clone the repository:
   ```bash
   git clone https://github.com/Mahaelango/Product_Recommentation_System.git
   ```
2. Navigate to the project folder:
   ```bash
   cd product-recommendation-system
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## **Dataset**
The dataset used in this project (`amazon.csv`) includes:
- Product name
- Category
- Ratings
- Review titles

You can customize the dataset by uploading your own CSV files.

## **Live Demo**

Check out the live version of this app here: [Streamlit App](https://approuctrecommentationsystem-mahaelango01.streamlit.app)

