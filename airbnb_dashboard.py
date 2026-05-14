

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


df = pd.read_csv("Cleaned_Data.csv")


df["service fee"] = (
    df["service fee"]
    .astype(str)
    .str.replace("[$, ]", "", regex=True)
    .astype(float)
)


df.dropna(subset=["neighbourhood group", "room type", "price", "cancellation_policy"], inplace=True)

print(f"   Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"   Neighbourhoods : {df['neighbourhood group'].nunique()}")
print(f"   Room types     : {df['room type'].nunique()}")
print(f"   Price range    : ${df['price'].min():,} – ${df['price'].max():,}")



PALETTE   = ["#FF5A5F", "#00A699", "#FC642D", "#484848", "#767676"]
BG_COLOR  = "#FAFAFA"
GRID_COLOR = "#E8E8E8"

sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor":   BG_COLOR,
    "axes.edgecolor":   "#CCCCCC",
    "axes.grid":        True,
    "grid.color":       GRID_COLOR,
    "grid.linewidth":   0.8,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
})



# DASHBOARD 1 — SUPPLY OVERVIEW  (2×2 grid)

fig = plt.figure(figsize=(16, 11))
fig.suptitle("Airbnb NYC — Supply Overview", fontsize=18, fontweight="bold",
             color="#484848", y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# 1a. Listings by Neighbourhood Group (bar) 
ax1 = fig.add_subplot(gs[0, 0])
nbhd_counts = df["neighbourhood group"].value_counts()
bars = ax1.bar(nbhd_counts.index, nbhd_counts.values, color=PALETTE, edgecolor="white", linewidth=0.8)
ax1.set_title("Listings by Neighbourhood Group")
ax1.set_xlabel("Neighbourhood Group")
ax1.set_ylabel("Number of Listings")
ax1.tick_params(axis="x", rotation=20)
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
             f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=9, color="#484848")

# 1b. Room Type Distribution (donut) 
ax2 = fig.add_subplot(gs[0, 1])
room_counts = df["room type"].value_counts()
wedges, texts, autotexts = ax2.pie(
    room_counts.values,
    labels=room_counts.index,
    autopct="%1.1f%%",
    colors=PALETTE,
    startangle=140,
    wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.5),
    pctdistance=0.75,
)
for at in autotexts:
    at.set_fontsize(9)
ax2.set_title("Room Type Distribution")

# 1c. Cancellation Policy Breakdown (horizontal bar) 
ax3 = fig.add_subplot(gs[1, 0])
cancel_counts = df["cancellation_policy"].value_counts()
ax3.barh(cancel_counts.index, cancel_counts.values, color=PALETTE[:len(cancel_counts)],
         edgecolor="white", linewidth=0.8)
ax3.set_title("Cancellation Policy Breakdown")
ax3.set_xlabel("Number of Listings")
for i, v in enumerate(cancel_counts.values):
    ax3.text(v + 100, i, f"{v:,}", va="center", fontsize=9, color="#484848")

# 1d. Host Identity Verified (stacked bar per neighbourhood) 
ax4 = fig.add_subplot(gs[1, 1])
identity_pivot = (
    df.groupby(["neighbourhood group", "host_identity_verified"])
    .size()
    .unstack(fill_value=0)
)
identity_pivot.plot(
    kind="bar", ax=ax4, color=["#FF5A5F", "#00A699"],
    edgecolor="white", linewidth=0.8, legend=True
)
ax4.set_title("Host Identity Verified by Area")
ax4.set_xlabel("Neighbourhood Group")
ax4.set_ylabel("Count")
ax4.tick_params(axis="x", rotation=25)
ax4.legend(title="Verified", fontsize=9)

plt.savefig("dashboard1_supply_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("Dashboard 1 saved → dashboard1_supply_overview.png")



# DASHBOARD 2 — PRICING ANALYSIS  

fig2 = plt.figure(figsize=(16, 11))
fig2.suptitle("Airbnb NYC — Pricing Analysis", fontsize=18, fontweight="bold",
              color="#484848", y=0.98)
gs2 = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.42, wspace=0.35)

# ── 2a. Price Distribution (histogram + KDE) ──
ax5 = fig2.add_subplot(gs2[0, 0])
price_capped = df[df["price"] < df["price"].quantile(0.95)]["price"]  # remove extreme outliers
sns.histplot(price_capped, bins=50, kde=True, color="#FF5A5F", alpha=0.7, ax=ax5,
             line_kws={"linewidth": 2})
ax5.set_title("Price Distribution (95th pct cap)")
ax5.set_xlabel("Price (USD)")
ax5.set_ylabel("Count")
ax5.axvline(price_capped.median(), color="#484848", linestyle="--", linewidth=1.4,
            label=f"Median ${price_capped.median():.0f}")
ax5.legend(fontsize=9)

# 2b. Avg Price by Neighbourhood Group 
ax6 = fig2.add_subplot(gs2[0, 1])
avg_price = df.groupby("neighbourhood group")["price"].mean().sort_values(ascending=False)
bars6 = ax6.bar(avg_price.index, avg_price.values, color=PALETTE, edgecolor="white")
ax6.set_title("Avg Price by Neighbourhood Group")
ax6.set_xlabel("Neighbourhood Group")
ax6.set_ylabel("Avg Price (USD)")
ax6.tick_params(axis="x", rotation=20)
for bar in bars6:
    ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             f"${bar.get_height():.0f}", ha="center", va="bottom", fontsize=9)

# 2c. Price by Room Type (box plot) 
ax7 = fig2.add_subplot(gs2[1, 0])
order = df.groupby("room type")["price"].median().sort_values(ascending=False).index
sns.boxplot(
    data=df[df["price"] < df["price"].quantile(0.95)],
    x="room type", y="price", order=order,
    palette=PALETTE, ax=ax7,
    flierprops=dict(marker="o", markersize=3, alpha=0.4),
)
ax7.set_title("Price Distribution by Room Type")
ax7.set_xlabel("Room Type")
ax7.set_ylabel("Price (USD)")
ax7.tick_params(axis="x", rotation=15)

# 2d. Price vs Service Fee (scatter)
ax8 = fig2.add_subplot(gs2[1, 1])
sample = df[df["price"] < df["price"].quantile(0.95)].sample(min(3000, len(df)), random_state=42)
scatter = ax8.scatter(
    sample["price"], sample["service fee"],
    c=sample["price"], cmap="RdYlGn_r",
    alpha=0.45, s=18, edgecolors="none"
)
fig2.colorbar(scatter, ax=ax8, label="Price (USD)")
ax8.set_title("Price vs. Service Fee")
ax8.set_xlabel("Price (USD)")
ax8.set_ylabel("Service Fee (USD)")

plt.savefig("dashboard2_pricing_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Dashboard 2 saved → dashboard2_pricing_analysis.png")



# DASHBOARD 3 — BOOKING BEHAVIOUR  

fig3 = plt.figure(figsize=(16, 11))
fig3.suptitle("Airbnb NYC — Booking Behaviour", fontsize=18, fontweight="bold",
              color="#484848", y=0.98)
gs3 = gridspec.GridSpec(2, 2, figure=fig3, hspace=0.42, wspace=0.35)

# ── 3a. Instant Bookable by Neighbourhood ──
ax9 = fig3.add_subplot(gs3[0, 0])
instant_pct = (
    df.groupby("neighbourhood group")["instant_bookable"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)
bars9 = ax9.bar(instant_pct.index, instant_pct.values, color=["#00A699"] * len(instant_pct),
                edgecolor="white")
ax9.set_title("Instant Bookable Rate by Neighbourhood (%)")
ax9.set_xlabel("Neighbourhood Group")
ax9.set_ylabel("% Instant Bookable")
ax9.tick_params(axis="x", rotation=20)
ax9.set_ylim(0, 100)
for bar in bars9:
    ax9.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

# 3b. Minimum Nights Distribution 
ax10 = fig3.add_subplot(gs3[0, 1])
min_nights_capped = df[df["minimum nights"] <= 30]["minimum nights"]
sns.histplot(min_nights_capped, bins=30, color="#FC642D", alpha=0.75, ax=ax10, kde=True,
             line_kws={"linewidth": 2})
ax10.set_title("Minimum Nights Distribution (≤30)")
ax10.set_xlabel("Minimum Nights")
ax10.set_ylabel("Count")

# 3c. Number of Reviews by Room Type (violin)
ax11 = fig3.add_subplot(gs3[1, 0])
reviews_capped = df[df["number of reviews"] <= df["number of reviews"].quantile(0.95)]
sns.violinplot(
    data=reviews_capped, x="room type", y="number of reviews",
    palette=PALETTE, ax=ax11, inner="quartile", cut=0
)
ax11.set_title("Reviews Distribution by Room Type")
ax11.set_xlabel("Room Type")
ax11.set_ylabel("Number of Reviews")
ax11.tick_params(axis="x", rotation=15)

# 3d. Construction Year Trend
ax12 = fig3.add_subplot(gs3[1, 1])
year_counts = (
    df["Construction year"]
    .dropna()
    .astype(int)
    .value_counts()
    .sort_index()
)
ax12.fill_between(year_counts.index, year_counts.values, alpha=0.25, color="#FF5A5F")
ax12.plot(year_counts.index, year_counts.values, color="#FF5A5F", linewidth=2.2, marker="o",
          markersize=5)
ax12.set_title("Listings by Construction Year")
ax12.set_xlabel("Year")
ax12.set_ylabel("Number of Listings")
ax12.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax12.tick_params(axis="x", rotation=30)

plt.savefig("dashboard3_booking_behaviour.png", dpi=150, bbox_inches="tight")
plt.show()
print("Dashboard 3 saved → dashboard3_booking_behaviour.png")



fig4, ax13 = plt.subplots(figsize=(12, 6))
fig4.suptitle("Airbnb NYC — Avg Price Heatmap\n(Neighbourhood Group × Room Type)",
              fontsize=16, fontweight="bold", color="#484848")

heatmap_data = (
    df.groupby(["neighbourhood group", "room type"])["price"]
    .mean()
    .unstack(fill_value=0)
)
sns.heatmap(
    heatmap_data, annot=True, fmt=".0f", cmap="YlOrRd",
    linewidths=0.5, linecolor="white",
    cbar_kws={"label": "Avg Price (USD)"},
    ax=ax13
)
ax13.set_title("")
ax13.set_xlabel("Room Type", fontsize=12)
ax13.set_ylabel("Neighbourhood Group", fontsize=12)
ax13.tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig("dashboard4_price_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Dashboard 4 saved → dashboard4_price_heatmap.png")


print("   Files saved in your working directory:")
print("   • dashboard1_supply_overview.png")
print("   • dashboard2_pricing_analysis.png")
print("   • dashboard3_booking_behaviour.png")
print("   • dashboard4_price_heatmap.png")
