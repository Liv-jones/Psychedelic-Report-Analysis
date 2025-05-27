# Erowid Trip Report Analysis

This project analyzes over 40,000 experiential reports from [Erowid.org](https://www.erowid.org/) to uncover sentiment and semantic patterns across 48 psychoactive substances. Techniques such as transformer-based sentiment analysis and latent semantic analysis (LSA) are used to interpret subjective experiences related to dissociatives, phenethylamines, tryptamines, lysergamides, and entactogens.

## Authors

- Olivia Jones, Carlos Mabrey, Casey Martin, & Jacy Werner
- Colorado State University – DSCI478
- Spring 2025

## Dataset

Trip reports were scraped from the [Erowid Center Wepsite](https://www.erowid.org/) using Selenium and BeautifulSoup, targeting metadata including:
  - Title
  - Substance(s)
  - Author
  - Dosage
  - Bodyweight
  - Report text and Date
  - Report sentiment and semantic content

## Installation

To ensure that the project runs properly, please install torch:

```!pip install torch```

  - The torch package is vital for sentiment analysis through its deep learning properties.

For webscraping, please install the following:

```selenium```

```beautifulsoup4```

Sentiment analysis was performed using a pre-trained transformer model:

```HuggingFace```

```spacesedan/sentiment-analysis-longformer```

  - This model is based on the longformer architecture, which is designed to efficiently process long text sequences.


## Methods

### 1. **Web Scraping**

  - Selenium was used for browser automation
  - BeautifulSoup parsed and extracted report contents

### 2. **Preprocessing**

  - Filtered to English and single-substance reports
  - Substances with < 15 reports were excluded

### 3. **Sentiment Analysis**

  - Model: spacesedan/sentiment-analysis-longformer
  - Categories: Very Negative, Negative, Neutral, Positive, Very Positive
  - Softmax: Applied to derive the probability distribution across sentiment categories

### 4. **Latent Semantic Analysis**

  - Preprocessing: lemmatization, tf-idf matrix creation using NLTK
  - Dimensionality reduction: SVD (top 20 components)
  - Similarity metrics: Pearson correlation and PCA
  - Clustering: K-means into 7 semantic themes

## Repository Contents

```Drug Assignments Grouped.csv```   – Assignment of drug categories to various group members.

```Drug List Grouped.csv```          – Grouping psychedelic drugs into five classes.

```Drug List.csv```                  - Full list of unorganized psychedelic drugs.

```LICENSE```                        - GNU General Public License v2.0 terms.

```READme.md```                      - Project Documentation.

```psychedelic_reports.zip```        - Zipped collection of raw trip reports

```selenium_webscraper.py```         - Script used to scrape trip reports from Erowid.org.

```sentiment_longformer_results.csv``` - Output of the sentiment model.

```words to exclude.csv```           - Hand picked words removed from trip reports.

## License

This project is licensed under the GNU General Public License v2.0.  
See the [LICENSE](LICENSE) file for more details.
