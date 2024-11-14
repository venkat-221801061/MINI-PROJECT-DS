I WANT YOU TO MAKE A SIMILAR IEEE PAPER FOR MY PROJECT: import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV, train_test_split
import warnings
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

class GigRadar:
    def __init__(self, freelancers_file, employers_file):
        print("Initializing GigRadar...")
        self.freelancers_df = pd.read_csv("/content/freelancers_dataset.csv")
        self.employers_df = pd.read_csv("/content/generated_employers_500.csv")
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.svd_model = SVD()
        self.content_based_matrix = None
        self.ratings_df = None
        self.surprise_dataset = None
        print("Initialization complete.\n")

    # Data Loading and Preprocessing Module
    def preprocess_data(self):
        print("Starting Data Loading and Preprocessing Module...")
        # Prepare data for collaborative filtering
        ratings_data = []
        for _, employer in self.employers_df.iterrows():
            employer_skills = set(employer['preferred_skills'].split(', '))
            for _, freelancer in self.freelancers_df.iterrows():
                freelancer_skills = set(freelancer['skills'].split(', '))
                skill_match = len(employer_skills.intersection(freelancer_skills))
                rating = min(5, max(1, skill_match + np.random.randint(0, 2)))
                ratings_data.append((employer['employer_id'], freelancer['freelancer_id'], rating))

        self.ratings_df = pd.DataFrame(ratings_data, columns=['employer_id', 'freelancer_id', 'rating'])
        print("Ratings data prepared.")

        # Create Surprise dataset
        reader = Reader(rating_scale=(1, 5))
        self.surprise_dataset = Dataset.load_from_df(self.ratings_df[['employer_id', 'freelancer_id', 'rating']], reader)
        print("Surprise dataset created.")

        # Prepare data for content-based filtering
        self.freelancers_df['skills_text'] = self.freelancers_df['skills'] + ' ' + self.freelancers_df['name']
        self.content_based_matrix = self.tfidf_vectorizer.fit_transform(self.freelancers_df['skills_text'])
        print("Content-based matrix created.")

        # Visualization: Rating Distribution
        plt.figure(figsize=(8,6))
        sns.countplot(x='rating', data=self.ratings_df)
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.show()
        print("Data Loading and Preprocessing Module completed.\n")

    # Collaborative Filtering Module
    def tune_collaborative_model(self):
        print("Starting Collaborative Filtering Module: Tuning Model...")
        param_grid = {
            'n_factors': [50, 100, 150],
            'n_epochs': [20, 30, 40],
            'lr_all': [0.002, 0.005],
            'reg_all': [0.02, 0.1]
        }

        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=-1)
        gs.fit(self.surprise_dataset)

        print("Best RMSE score: ", gs.best_score['rmse'])
        print("Best configuration: ", gs.best_params['rmse'])

        # Update the model with best parameters
        best_params = gs.best_params['rmse']
        self.svd_model = SVD(n_factors=best_params['n_factors'],
                             n_epochs=best_params['n_epochs'],
                             lr_all=best_params['lr_all'],
                             reg_all=best_params['reg_all'])
        print("Collaborative Filtering Model tuning completed.\n")

    def train_collaborative_model(self):
        print("Training Collaborative Filtering Model...")
        trainset = self.surprise_dataset.build_full_trainset()
        self.svd_model.fit(trainset)
        print("Collaborative Filtering Model training completed.\n")

    def evaluate_collaborative_model(self):
        print("Evaluating Collaborative Filtering Model...")
        # Split the data into training and testing sets
        trainset, testset = train_test_split(self.surprise_dataset, test_size=0.25)

        # Train the model
        self.svd_model.fit(trainset)

        # Make predictions on the test set
        predictions = self.svd_model.test(testset)

        # Convert predictions to binary classifications (1 if rating >= 3, 0 otherwise)
        true_ratings = [1 if pred.r_ui >= 3 else 0 for pred in predictions]
        predicted_ratings = [1 if pred.est >= 3 else 0 for pred in predictions]

        # Calculate metrics
        accuracy = accuracy_score(true_ratings, predicted_ratings)
        precision = precision_score(true_ratings, predicted_ratings)
        recall = recall_score(true_ratings, predicted_ratings)
        f1 = f1_score(true_ratings, predicted_ratings)

        # Calculate RMSE and MAE
        rmse = np.sqrt(np.mean([(pred.r_ui - pred.est) ** 2 for pred in predictions]))
        mae = np.mean([abs(pred.r_ui - pred.est) for pred in predictions])

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        # Visualization: RMSE and MAE
        metrics = {'RMSE': rmse, 'MAE': mae}
        names = list(metrics.keys())
        values = list(metrics.values())

        plt.figure(figsize=(6,4))
        sns.barplot(x=names, y=values)
        plt.title('Model Evaluation Metrics')
        plt.ylabel('Score')
        plt.show()
        print("Collaborative Filtering Model evaluation completed.\n")

    # Content Based Filtering Module
    def get_content_based_recommendations(self, employer_id, n=5):
        print(f"Generating Content-Based Recommendations for Employer ID: {employer_id}...")
        employer_skills = self.employers_df.loc[self.employers_df['employer_id'] == employer_id, 'preferred_skills'].iloc[0]
        employer_vector = self.tfidf_vectorizer.transform([employer_skills])
        similarities = cosine_similarity(employer_vector, self.content_based_matrix).flatten()
        similar_indices = similarities.argsort()[::-1]
        recommendations = self.freelancers_df.iloc[similar_indices[:n]]['freelancer_id'].tolist()
        print(f"Content-Based Recommendations generated.\n")
        return recommendations

    # Hybrid Recommendation Module
    def get_collaborative_recommendations(self, employer_id, n=5):
        print(f"Generating Collaborative Filtering Recommendations for Employer ID: {employer_id}...")
        employer_ratings = self.ratings_df[self.ratings_df['employer_id'] == employer_id]
        unrated_freelancers = self.freelancers_df[~self.freelancers_df['freelancer_id'].isin(employer_ratings['freelancer_id'])]

        def predict_rating(freelancer):
            return (freelancer['freelancer_id'], self.svd_model.predict(employer_id, freelancer['freelancer_id']).est)

        with ThreadPoolExecutor() as executor:
            predictions = list(executor.map(predict_rating, [row for _, row in unrated_freelancers.iterrows()]))

        predictions.sort(key=lambda x: x[1], reverse=True)
        recommendations = [p[0] for p in predictions[:n]]
        print(f"Collaborative Filtering Recommendations generated.\n")
        return recommendations

    def get_hybrid_recommendations(self, employer_id, n=5):
        print(f"Generating Hybrid Recommendations for Employer ID: {employer_id}...")
        collaborative_recs = self.get_collaborative_recommendations(employer_id, n)
        content_based_recs = self.get_content_based_recommendations(employer_id, n)

        # Combine and weight recommendations
        hybrid_recs = {}
        for i, rec in enumerate(collaborative_recs):
            hybrid_recs[rec] = hybrid_recs.get(rec, 0) + (n - i) * 0.6
        for i, rec in enumerate(content_based_recs):
            hybrid_recs[rec] = hybrid_recs.get(rec, 0) + (n - i) * 0.4

        sorted_recs = sorted(hybrid_recs.items(), key=lambda x: x[1], reverse=True)
        final_recommendations = [rec[0] for rec in sorted_recs[:n]]
        print(f"Hybrid Recommendations generated.\n")
        return final_recommendations

    # User Interface and Recommendation Generation Module
    def get_matching_employers(self, freelancer_id, n=5):
        print(f"Finding Matching Employers for Freelancer ID: {freelancer_id}...")
        freelancer_skills = self.freelancers_df.loc[self.freelancers_df['freelancer_id'] == freelancer_id, 'skills'].iloc[0]
        freelancer_vector = self.tfidf_vectorizer.transform([freelancer_skills])
        employer_vectors = self.tfidf_vectorizer.transform(self.employers_df['preferred_skills'])
        similarities = cosine_similarity(freelancer_vector, employer_vectors).flatten()
        similar_indices = similarities.argsort()[::-1]
        recommendations = self.employers_df.iloc[similar_indices[:n]]['employer_id'].tolist()
        print(f"Matching Employers found.\n")
        return recommendations

    def test_new_data(self, new_data, user_type='employer'):
        if user_type == 'employer':
            print("Testing new employer data for recommendations...")
            new_vector = self.tfidf_vectorizer.transform([new_data['preferred_skills']])
            similarities = cosine_similarity(new_vector, self.content_based_matrix).flatten()
            similar_indices = similarities.argsort()[::-1]
            similar_freelancers = self.freelancers_df.iloc[similar_indices[:5]]
            print("Recommendations for new employer generated.\n")
            return similar_freelancers[['freelancer_id', 'name', 'skills', 'rating']].to_dict('records')
        elif user_type == 'freelancer':
            print("Testing new freelancer data for recommendations...")
            new_vector = self.tfidf_vectorizer.transform([new_data['skills']])
            employer_vectors = self.tfidf_vectorizer.transform(self.employers_df['preferred_skills'])
            similarities = cosine_similarity(new_vector, employer_vectors).flatten()
            similar_indices = similarities.argsort()[::-1]
            similar_employers = self.employers_df.iloc[similar_indices[:5]]
            print("Recommendations for new freelancer generated.\n")
            return similar_employers[['employer_id', 'company_name', 'preferred_skills']].to_dict('records')

    # Additional Visualization Methods
    def visualize_jobs_posted(self):
        print("Visualizing Jobs Posted by Employers...")
        plt.figure(figsize=(10,6))
        sns.histplot(self.employers_df['jobs_posted'], bins=20, kde=True)
        plt.title('Distribution of Jobs Posted by Employers')
        plt.xlabel('Jobs Posted')
        plt.ylabel('Number of Employers')
        plt.show()
        print("Jobs Posted visualization completed.\n")

    def visualize_freelancer_skills(self):
        print("Visualizing Freelancer Skills Distribution...")
        all_skills = self.freelancers_df['skills'].str.split(', ').explode()
        plt.figure(figsize=(12,8))
        sns.countplot(y=all_skills, order=all_skills.value_counts().index)
        plt.title('Freelancer Skills Distribution')
        plt.xlabel('Count')
        plt.ylabel('Skills')
        plt.show()
        print("Freelancer Skills Distribution visualization completed.\n")

def main():
    gigradar = GigRadar('freelancers.csv', 'employers.csv')

    # Data Loading and Preprocessing Module
    gigradar.preprocess_data()

    # Visualizations
    gigradar.visualize_jobs_posted()
    gigradar.visualize_freelancer_skills()

    # Collaborative Filtering Module
    print("=== Collaborative Filtering Module ===")
    gigradar.tune_collaborative_model()
    gigradar.train_collaborative_model()
    gigradar.evaluate_collaborative_model()

    while True:
        print("\nWelcome to GigRadar!")
        print("1. Employer")
        print("2. Freelancer")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            print("\nEmployer Menu:")
            print("1. Get recommendations for existing employer")
            print("2. Get recommendations for new employer")
            emp_choice = input("Enter your choice (1/2): ")

            if emp_choice == '1':
                employer_id = input("Enter employer ID: ")
                if employer_id in gigradar.employers_df['employer_id'].astype(str).values:
                    recommendations = gigradar.get_hybrid_recommendations(employer_id)
                    print(f"\nRecommended freelancers for employer {employer_id}:")
                    for rec in recommendations:
                        freelancer = gigradar.freelancers_df[gigradar.freelancers_df['freelancer_id'] == int(rec)].iloc[0]
                        print(f"Freelancer ID: {rec}, Name: {freelancer['name']}, Skills: {freelancer['skills']}")
                else:
                    print("Employer ID not found. Please try again.")

            elif emp_choice == '2':
                skills = input("Enter preferred skills (comma-separated): ")
                new_employer = {'preferred_skills': skills}
                results = gigradar.test_new_data(new_employer, 'employer')
                print("\nMatching freelancers for new employer:")
                for freelancer in results:
                    print(f"Freelancer ID: {freelancer['freelancer_id']}, Name: {freelancer['name']}, Skills: {freelancer['skills']}, Rating: {freelancer['rating']}")

        elif choice == '2':
            print("\nFreelancer Menu:")
            print("1. Get matching employers for existing freelancer")
            print("2. Get matching employers for new freelancer")
            free_choice = input("Enter your choice (1/2): ")

            if free_choice == '1':
                freelancer_id = input("Enter freelancer ID: ")
                if int(freelancer_id) in gigradar.freelancers_df['freelancer_id'].values:
                    matching_employers = gigradar.get_matching_employers(int(freelancer_id))
                    print(f"\nMatching employers for freelancer {freelancer_id}:")
                    for emp in matching_employers:
                        employer = gigradar.employers_df[gigradar.employers_df['employer_id'] == emp].iloc[0]
                        print(f"Employer ID: {emp}, Company: {employer['company_name']}, Preferred Skills: {employer['preferred_skills']}")
                else:
                    print("Freelancer ID not found. Please try again.")

            elif free_choice == '2':
                skills = input("Enter your skills (comma-separated): ")
                new_freelancer = {'skills': skills}
                results = gigradar.test_new_data(new_freelancer, 'freelancer')
                print("\nMatching employers for new freelancer:")
                for employer in results:
                    print(f"Employer ID: {employer['employer_id']}, Company: {employer['company_name']}, Preferred Skills: {employer['preferred_skills']}")

        elif choice == '3':
            print("Thank you for using GigRadar. Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
