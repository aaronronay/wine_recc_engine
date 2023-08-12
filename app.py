import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class WineRecommender:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.engine_data = None
        self.tfidf_matrix = None
        self.cosine_similarities = None
        self.results = {}

    def prepare_engine_data(self):
        self.engine_data = self.data[['title', 'variety', 'points', 'price', 'taster_name', 'description']].dropna()

    def create_tfidf_matrix(self):
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
        self.tfidf_matrix = tf.fit_transform(self.engine_data['description'])

    def calculate_cosine_similarities(self):
        self.cosine_similarities = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)

    def generate_recommendations(self):
        for idx, _ in self.engine_data.iterrows():
            similar_indices = self.cosine_similarities[idx].argsort()[:-100:-1]
            self.results[idx] = [(self.cosine_similarities[idx][i], i) for i in similar_indices][1:]

    def item(self, id):
        return self.engine_data.loc[id, 'title']

    def recommender(self, id, num):
        print(f"Recommending {num} wines similar to {self.item(id)}\n-------")
        try:
            recs = self.results[id][:num]
            for rec in recs:
                print(f"Recommended: {self.item(rec[1])} (score: {rec[0]})")
        except KeyError:
            print('Oops, key error. Try mapping the dictionary and tune the function to access the correct key.')

    def process_wine_titles_file(self, input_file, output_file, num_recommendations):
        with open(input_file, 'r') as f:
            wine_titles = f.read().splitlines()

        recommendations = {}

        for title in wine_titles:
            matching_row = self.engine_data[self.engine_data['title'].str.contains(title)]
            wine_id = matching_row.index[0] if not matching_row.empty else None
            recommendations[title] = self.results.get(wine_id, []) if wine_id is not None else []

        with open(output_file, 'w') as f:
            for key, recs in recommendations.items():
                if recs:
                    f.write(f"Recommendations for wine '{self.item(key)}' (ID: {key}):\n")
                    for rec in recs:
                        rec_title = self.item(rec[1])
                        f.write(f"Recommended: {rec_title} (score: {rec[0]})\n")
                    f.write("\n")
                else:
                    f.write(f"No recommendations found for wine '{key}'\n\n")

def main():
    csv_file_path = 'wine_data_2022.csv'
    input_file_path = 'wine_titles.txt'
    output_file_path = 'recommendations_output.txt'
    num_recommendations = 5

    wine_recommender = WineRecommender(csv_file_path)
    wine_recommender.prepare_engine_data()
    wine_recommender.create_tfidf_matrix()
    wine_recommender.calculate_cosine_similarities()
    wine_recommender.generate_recommendations()
    wine_recommender.process_wine_titles_file(input_file_path, output_file_path, num_recommendations)

if __name__ == "__main__":
    main()
