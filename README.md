# Wine Recommender

This Python script provides a wine recommendation system based on wine descriptions using the TF-IDF technique.

## Description

This script uses the WineRecommender class to process wine data from a CSV file, calculate similarity scores using TF-IDF vectors, and generate wine recommendations based on input wine titles. The recommendations are then saved to an output text file.

## Requirements

- Python 3.x
- pandas
- scikit-learn

## Usage

1. Place your CSV file containing wine data (e.g., `wine_data_2022.csv`) in the same directory as the script.
2. Create a text file (e.g., `wine_titles.txt`) containing a list of wine titles you want to generate recommendations for, with each title on a new line.
3. Run the script using the command: `python script_name.py`

## Configuration

- `csv_file_path`: Path to the CSV file containing wine data.
- `input_file_path`: Path to the text file containing wine titles for recommendations.
- `output_file_path`: Path to the output text file where recommendations will be saved.
- `num_recommendations`: Number of recommendations to generate for each wine title.

## Result

The script will generate recommendations for the wine titles specified in the input file and save the results to the output text file.

## Acknowledgments

The TF-IDF technique used for similarity calculation is inspired by information retrieval and text analysis concepts.

## License

This project is licensed under the [MIT License](LICENSE).
