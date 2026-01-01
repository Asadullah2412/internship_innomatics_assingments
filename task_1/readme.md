## Exploratory Data Analysis (EDA)

### Dataset Overview

The dataset consists of user performance scores grouped by batch. After initial cleaning and preprocessing, the following columns were used for analysis:

* **batch**: Batch identifier (e.g., AI_ELITE_4, AI_ELITE_6, AI_ELITE_7)
* **user_id**: Unique identifier for each user
* **score_int**: Integer score extracted from the original string-based score column (e.g., `6/7 → 6`)

The primary goal of this EDA was to understand **how user performance varies across different batches**.

---

### Score Distribution Across Batches

To compare performance patterns, a batch-wise score distribution plot was created.

**Key observations:**

* **AI_ELITE_7**

  * Scores are heavily concentrated in the higher range (5–7).
  * Very few low scores are observed.
  * Indicates strong and consistent performance across users.

* **AI_ELITE_6**

  * Scores are widely spread across the entire range.
  * Presence of both low and high performers.
  * Suggests higher variability and inconsistent outcomes within the batch.

* **AI_ELITE_4**

  * Majority of scores cluster around the mid-range (score = 4).
  * Fewer high scorers compared to AI_ELITE_7.
  * Reflects average overall performance with limited top-end results.

---

### Insights

* Batch performance differs not only in terms of average scores but also in **consistency and score distribution**.
* **AI_ELITE_7** emerges as the strongest performing batch, combining higher scores with lower variability.
* **AI_ELITE_6** shows potential but lacks consistency, indicating a possible need for targeted interventions.
* **AI_ELITE_4** maintains stable mid-level performance but may require strategies to encourage high achievers.

---

### Conclusion

This EDA highlights the importance of analyzing score distributions rather than relying solely on summary statistics like mean. Visual comparison across batches provides deeper insights into performance patterns, consistency, and batch effectiveness, which can guide further analysis or modeling steps.
