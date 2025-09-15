# Models for Movie Recommendations

This project demonstrates matrix factorization techniques for movie recommendation using the MovieLens 100k dataset. All code and analysis are contained in the Jupyter notebook `matrix_factorization.ipynb`.

## Features
- Loads and preprocesses MovieLens data
- Applies SVD and ALS for collaborative filtering
- Finds similar movies and top recommendations for users

## Usage
1. Clone or download this repository.
2. Place the MovieLens 100k data in the `data/ml-100k/` directory (CSV format).
3. Open `matrix_factorization.ipynb` in Jupyter Notebook or VS Code.
4. Run the notebook cells to reproduce results and explore recommendations.

## Requirements
- Python 3.8+
- pandas, numpy, scikit-learn, scipy


## Data Source
- [MovieLens 100k Dataset](https://grouplens.org/datasets/movielens/latest/)

## Output
- Top-N movie recommendations for a user
- Most similar movies to a given title

---