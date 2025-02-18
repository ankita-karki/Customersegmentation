#  Customer Segmentation using RFM Analysis and K-means Clustering

This project performs customer segmentation on the **Retail Sales Dataset** extracted from [Kaggle](https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset). The analysis uses **RFM (Recency, Frequency, Monetary)** metrics combined with **K-means clustering** to identify distinct customer groups. These insights can help businesses tailor marketing strategies to improve customer engagement and increase ROI.

---

## Dataset Overview

The dataset contains transactional data for a retail business, including details such as:
- **Transaction ID**: Unique identifier for each transaction.
- **Date**: Date of the transaction.
- **Customer ID**: Unique identifier for each customer.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Product Category**: Category of the purchased product.
- **Quantity**: Number of units purchased.
- **Price per Unit**: Price of one unit of the product.
- **Total Amount**: Total monetary value of the transaction.

Source: [Kaggle - Retail Sales Dataset](https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset)

---

## Objective

The primary goal of this project is to:
- Perform **customer segmentation** using RFM analysis and K-means clustering.
- Identify high-value customer groups and provide actionable insights for targeted marketing strategies.
- Visualize the results to facilitate decision-making.

---

## Methodology

1. **Data Cleaning**:
   - Handle missing values in critical columns (`Customer ID`, `Date`, `Total Amount`).
   - Convert the `Date` column to a datetime format.

2. **RFM Calculation**:
   - Compute **Recency**, **Frequency**, and **Monetary** metrics for each customer.
   - Define the snapshot date as one day after the most recent transaction date.

3. **Normalization and Scaling**:
   - Normalize the RFM metrics to ensure they have a mean of 0 and standard deviation of 1.
   - Apply log transformation to reduce skewness in the `Monetary` metric.
   - Scale the normalized data using `StandardScaler`.

4. **Clustering**:
   - Use the **Elbow Method** to determine the optimal number of clusters (`K`).
   - Perform K-means clustering with the chosen `K` value.

5. **Visualization**:
   - Use PCA (Principal Component Analysis) to reduce the dimensionality of the data for visualization.
   - Plot the clusters in a 2D scatter plot.

6. **Segment Labeling**:
   - Assign meaningful labels to each cluster based on their RFM characteristics (e.g., "High-Value Loyalists", "At-Risk Customers").

---

## Code Workflow

1. Load the dataset and inspect its structure.
2. Clean the data by handling missing values and ensuring proper data types.
3. Calculate RFM metrics (`Recency`, `Frequency`, `Monetary`) for each customer.
4. Normalize and scale the RFM data.
5. Use the Elbow Method to determine the optimal number of clusters.
6. Perform K-means clustering and assign cluster labels.
7. Generate a summary of each cluster's characteristics.
8. Visualize the clusters using PCA.
9. Label the clusters for actionable insights.

---

## Results

- **Cluster Summary**:
  A table summarizing the average Recency, Frequency, and Monetary values for each cluster, along with the count of customers in each cluster.

- **Visualizations**:
  - Histograms showing the distribution of RFM metrics.
  - Scatter plot visualizing the clusters in two dimensions using PCA.

- **Segment Labels**:
  Each cluster is assigned a label based on its RFM characteristics:
  - **High-Value Loyalists**: Customers with high Frequency and Monetary values.
  - **At-Risk Customers**: Customers with low Recency, low Frequency, and low Monetary values.
  - **Potential Loyalists**: Customers with moderate RFM values.
  - **Low-Engagement**: Customers with high Recency and low Frequency/Monetary values.

---

## Dependencies

To run this project, you need the following Python libraries:

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
```

You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mohammadtalib786/retail-sales-dataset) and save it as `retail_sales_dataset.csv`.
2. Place the dataset in a folder named `Data/` in the same directory as the script.
3. Run the script using Python:

```bash
python customer_segmentation.py
```

---

## Contributing

Feel free to contribute to this project by suggesting improvements or adding new features. To contribute:
1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request with a clear description of your modifications.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

--- 

If you have any questions or need further clarification, feel free to reach out!
