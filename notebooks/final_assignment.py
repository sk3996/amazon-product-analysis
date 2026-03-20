# %%
# %% [markdown]
# # Final Assignment
# ## Understanding Product Performance in an Amazon Sales Dataset
#
# **Name:** Simar Kaur
#
# ### Research Questions
# 1. Which product categories perform best?
# 2. Are ratings and review counts linked to higher sales?

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# load raw data
df = pd.read_csv("amazon (1).csv")

print("Raw shape:", df.shape)
display(df.head())

# %%
# clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
)

print(df.columns.tolist())

# %%
# basic cleaning and type fixes
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

print("Duplicate rows before drop:", df.duplicated().sum())
df = df.drop_duplicates().copy()
print("Duplicate rows after drop:", df.duplicated().sum())

# %%
# missing values
missing_summary = df.isna().sum().sort_values(ascending=False)
print("Missing values by column:")
display(missing_summary)

df_clean = df.dropna().copy()
print("Clean shape:", df_clean.shape)

# %%
# feature engineering
df_clean["order_year"] = df_clean["order_date"].dt.year
df_clean["order_month"] = df_clean["order_date"].dt.month
df_clean["order_day"] = df_clean["order_date"].dt.day
df_clean["order_dayofweek"] = df_clean["order_date"].dt.dayofweek
df_clean["order_quarter"] = df_clean["order_date"].dt.quarter

df_clean["discount_rate"] = df_clean["discount_percent"] / 100
df_clean["computed_discounted_price"] = df_clean["price"] * (1 - df_clean["discount_rate"])
df_clean["computed_total_revenue"] = df_clean["computed_discounted_price"] * df_clean["quantity_sold"]
df_clean["is_discounted"] = (df_clean["discount_percent"] > 0).astype(int)

df_clean["payment_method_clean"] = (
    df_clean["payment_method"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.title()
)

# %%
# focused dataframe for final project
project_df = df_clean[
    [
        "product_id",
        "product_category",
        "customer_region",
        "payment_method_clean",
        "price",
        "discount_percent",
        "quantity_sold",
        "rating",
        "review_count",
        "total_revenue",
        "order_date",
        "order_month",
        "order_year"
    ]
].copy()

display(project_df.head())
project_df.info()

# %%
# descriptive statistics
display(project_df.describe(include="all"))

print("Number of rows:", len(project_df))
print("Number of columns:", project_df.shape[1])
print("Unique product categories:", project_df["product_category"].nunique())
print("Unique regions:", project_df["customer_region"].nunique())
print("Date range:", project_df["order_date"].min(), "to", project_df["order_date"].max())

# %% [markdown]
# ## Research Question 1
# ### Which product categories perform best?

# %%
category_summary = (
    project_df.groupby("product_category")
    .agg(
        total_revenue=("total_revenue", "sum"),
        total_quantity_sold=("quantity_sold", "sum"),
        avg_price=("price", "mean"),
        avg_rating=("rating", "mean"),
        avg_review_count=("review_count", "mean"),
        order_count=("product_category", "size")
    )
    .sort_values("total_revenue", ascending=False)
    .reset_index()
)

display(category_summary)

# %%
top_category = category_summary.iloc[0]
bottom_category = category_summary.iloc[-1]

print("Top category by revenue:", top_category["product_category"], round(top_category["total_revenue"], 2))
print("Lowest category by revenue:", bottom_category["product_category"], round(bottom_category["total_revenue"], 2))

# %%
# figure 1: total revenue by category
plt.figure(figsize=(10, 6))
plt.bar(category_summary["product_category"], category_summary["total_revenue"])
plt.title("Total Revenue by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../figures/figure_1_total_revenue_by_category.png", dpi=300, bbox_inches="tight")
plt.show()


# figure 2: average revenue per order by category
category_avg_rev = (
    project_df.groupby("product_category")["total_revenue"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

plt.figure(figsize=(10, 6))
plt.bar(category_avg_rev["product_category"], category_avg_rev["total_revenue"])
plt.title("Average Revenue per Order by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Average Revenue per Order")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../figures/figure_2_avg_revenue_by_category.png", dpi=300, bbox_inches="tight")
plt.show()


# pivot table: category x region revenue
region_pivot = pd.pivot_table(
    project_df,
    values="total_revenue",
    index="product_category",
    columns="customer_region",
    aggfunc="sum"
)

display(region_pivot)

# %%
# figure 3: heatmap-style plot using matplotlib only
plt.figure(figsize=(10, 6))
plt.imshow(region_pivot, aspect="auto")
plt.colorbar(label="Total Revenue")
plt.title("Total Revenue by Product Category and Region")
plt.xlabel("Customer Region")
plt.ylabel("Product Category")
plt.xticks(ticks=np.arange(len(region_pivot.columns)), labels=region_pivot.columns, rotation=45)
plt.yticks(ticks=np.arange(len(region_pivot.index)), labels=region_pivot.index)
plt.tight_layout()
plt.savefig("../figures/figure_3_revenue_heatmap_category_region.png", dpi=300, bbox_inches="tight")
plt.show()


# ## Research Question 2
# ### Are ratings and review counts linked to higher sales?

# %%
sales_corr = project_df[
    ["rating", "review_count", "quantity_sold", "total_revenue", "price", "discount_percent"]
].corr(numeric_only=True)

display(sales_corr)


# figure 4: scatter of quantity sold vs revenue
sample_n = min(8000, len(project_df))
plot_df = project_df.sample(sample_n, random_state=0).copy()

plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    plot_df["quantity_sold"],
    plot_df["total_revenue"],
    c=plot_df["rating"],
    alpha=0.35
)
plt.colorbar(scatter, label="Rating")
plt.title("Total Revenue vs Quantity Sold, Colored by Rating")
plt.xlabel("Quantity Sold")
plt.ylabel("Total Revenue")
plt.tight_layout()
plt.savefig("../figures/figure_4_revenue_vs_quantity_rating.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# figure 5: product-level bubble chart
product_perf = (
    project_df.groupby("product_id")
    .agg(
        avg_rating=("rating", "mean"),
        avg_reviews=("review_count", "mean"),
        total_sales=("total_revenue", "sum")
    )
    .reset_index()
)

plt.figure(figsize=(9, 6))
plt.scatter(
    product_perf["avg_rating"],
    product_perf["avg_reviews"],
    s=np.maximum(product_perf["total_sales"] / 50, 10),
    alpha=0.4
)
plt.title("Product Performance: Rating, Reviews, and Sales")
plt.xlabel("Average Rating")
plt.ylabel("Average Review Count")
plt.tight_layout()
plt.savefig("../figures/figure_5_product_bubble_chart.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
# figure 6: review bins vs average revenue
project_df["review_bin"] = pd.qcut(
    project_df["review_count"],
    q=10,
    duplicates="drop"
)

trend = (
    project_df.groupby("review_bin", observed=False)["total_revenue"]
    .mean()
    .reset_index()
)

trend["bin_mid"] = trend["review_bin"].apply(lambda x: x.mid)

plt.figure(figsize=(9, 6))
plt.plot(trend["bin_mid"], trend["total_revenue"], marker="o")
plt.title("Average Total Revenue by Review Count")
plt.xlabel("Review Count Bin Midpoint")
plt.ylabel("Average Total Revenue per Order")
plt.tight_layout()
plt.savefig("../figures/figure_6_review_bins_vs_revenue.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
#used concepts from my econometrics class to do some statistical analysis
# simple correlation summary
corr_rating_revenue = project_df["rating"].corr(project_df["total_revenue"])
corr_reviews_revenue = project_df["review_count"].corr(project_df["total_revenue"])
corr_rating_quantity = project_df["rating"].corr(project_df["quantity_sold"])
corr_reviews_quantity = project_df["review_count"].corr(project_df["quantity_sold"])

print("Correlation: rating vs total_revenue =", round(corr_rating_revenue, 4))
print("Correlation: review_count vs total_revenue =", round(corr_reviews_revenue, 4))
print("Correlation: rating vs quantity_sold =", round(corr_rating_quantity, 4))
print("Correlation: review_count vs quantity_sold =", round(corr_reviews_quantity, 4))

# %%
# simple OLS using numpy only
reg_df = project_df[["rating", "review_count", "quantity_sold", "total_revenue"]].dropna().copy()

X = reg_df[["rating", "review_count"]].to_numpy()
X = np.column_stack([np.ones(len(X)), X])
y = reg_df["total_revenue"].to_numpy()

beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
y_hat = X @ beta
resid = y - y_hat

n = len(y)
k = X.shape[1]
sse = np.sum(resid ** 2)
sst = np.sum((y - y.mean()) ** 2)
r2 = 1 - sse / sst

sigma2 = sse / (n - k)
var_beta = sigma2 * np.linalg.inv(X.T @ X)
se_beta = np.sqrt(np.diag(var_beta))

coef_table = pd.DataFrame({
    "term": ["intercept", "rating", "review_count"],
    "coefficient": beta,
    "std_error": se_beta
})

display(coef_table)
print("R-squared:", round(r2, 4))

# %%
# second regression with quantity sold as dependent variable
X2 = reg_df[["rating", "review_count"]].to_numpy()
X2 = np.column_stack([np.ones(len(X2)), X2])
y2 = reg_df["quantity_sold"].to_numpy()

beta2 = np.linalg.inv(X2.T @ X2) @ (X2.T @ y2)
y2_hat = X2 @ beta2
resid2 = y2 - y2_hat

sse2 = np.sum(resid2 ** 2)
sst2 = np.sum((y2 - y2.mean()) ** 2)
r2_2 = 1 - sse2 / sst2

sigma2_2 = sse2 / (len(y2) - X2.shape[1])
var_beta2 = sigma2_2 * np.linalg.inv(X2.T @ X2)
se_beta2 = np.sqrt(np.diag(var_beta2))

coef_table_2 = pd.DataFrame({
    "term": ["intercept", "rating", "review_count"],
    "coefficient": beta2,
    "std_error": se_beta2
})

display(coef_table_2)
print("R-squared:", round(r2_2, 4))

# %% [markdown]
# ## Additional Table for Business Context

# %%
payment_crosstab = pd.crosstab(
    project_df["product_category"],
    project_df["payment_method_clean"]
)

display(payment_crosstab)

# %% [markdown]
# ## Final Summary

# %%


# %%
# save final outputs
project_df.to_csv("final_project_dataset.csv", index=False)
category_summary.to_csv("category_summary.csv", index=False)
region_pivot.to_csv("category_region_revenue_pivot.csv")

print("Files saved successfully.")


