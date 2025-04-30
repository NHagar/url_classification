import duckdb
import matplotlib.pyplot as plt
import seaborn as sns

con = duckdb.connect(database=":memory:", read_only=False)
df = con.execute(
    "SELECT model, dataset, throughput FROM './data/processed/evaluation_metrics_with_variants_throughput.csv' WHERE text_variant IS NULL AND url_variant IS NULL"
).fetchdf()

p = sns.barplot(data=df, x="model", y="throughput")

plt.yscale("log")

plt.xlabel("Model")
plt.ylabel("Throughput (log scale)")


print(df.groupby("model")["throughput"].mean())

# save plot
plt.savefig("./data/processed/throughput.png")

plt.show()
