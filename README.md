# Task 3: Bank Marketing Dataset Exploration and Analysis

## 📌 Objective

The aim of this task is to perform exploratory data analysis (EDA) on the **Bank Marketing Dataset** to understand the behavior of customers and their likelihood of subscribing to a term deposit. The analysis includes identifying key features that influence customer decisions and patterns.

---

## 📁 Dataset Description

The dataset is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. It includes attributes for customer profiles and campaign outcomes.

### 🔑 Key Columns:

| Column | Description |
|--------|-------------|
| `age` | Age of the client |
| `job` | Job type (admin, technician, etc.) |
| `marital` | Marital status |
| `education` | Education level |
| `default` | Has credit in default? (yes/no) |
| `balance` | Average yearly balance in euros |
| `housing` | Has housing loan? (yes/no) |
| `loan` | Has personal loan? (yes/no) |
| `contact` | Contact communication type |
| `day`, `month` | Last contact day and month |
| `duration` | Last contact duration (in seconds) |
| `campaign` | Number of contacts during the campaign |
| `pdays` | Days since last contact |
| `previous` | Number of contacts before this campaign |
| `poutcome` | Outcome of the previous marketing campaign |
| `y` | Target variable – has the client subscribed to a term deposit? (yes/no) |

---

## 🔍 Exploratory Data Analysis (EDA)

Performed the following tasks:

1. **Data Cleaning**  
   - Checked for missing values  
   - Handled categorical encoding where necessary

2. **Univariate Analysis**  
   - Plotted distributions of age, balance, duration, etc.  
   - Count plots for categorical features like job, marital, education

3. **Bivariate Analysis**  
   - Relationship between age, job, balance vs target (`y`)  
   - Impact of previous campaign outcomes on subscription

4. **Target Variable Analysis**  
   - Subscribed (`y`) vs not subscribed  
   - Influential factors visualized using bar plots and box plots

5. **Correlation Analysis**  
   - Heatmap to show feature correlations  
   - Insights on features that strongly correlate with the target

---

## 📦 Tools Used

- Python
- Pandas
- Seaborn
- Matplotlib
- VS Code or Jupyter Notebook

contact me:www.linkedin.com/in/poornisha-krishnan-0579802a6
