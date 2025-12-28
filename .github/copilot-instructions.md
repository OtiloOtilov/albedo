# Albedo: GitHub Repo Recommender System

## Architecture Overview

Albedo is a two-stage ML recommender system: a **Scala/Spark ML pipeline** for offline model training and feature engineering, plus a **Django/Python web layer** for data collection and serving recommendations.

### Component Boundaries

- **Spark ML Pipeline** (`src/main/scala/ws/vinta/albedo/`): Batch processing for feature engineering, model training, and evaluation
  - Profile builders: `UserProfileBuilder.scala`, `RepoProfileBuilder.scala` - extract features from raw data
  - Recommender builders: `ALSRecommenderBuilder.scala`, `ContentRecommenderBuilder.scala`, `LogisticRegressionRanker.scala`
  - Custom transformers in `transformers/`: `NegativeBalancer`, `HanLPTokenizer`, `SnowballStemmer`
  - Custom recommenders in `recommenders/`: All extend abstract `Recommender` class (Spark ML Transformer pattern)
- **Django Backend** (`app/`, `albedo/`): Data collection via GitHub API, model metadata storage
  - Models: `UserInfo`, `RepoInfo`, `RepoStarring`, `UserRelation` in `app/models.py`
  - Management commands: `collect_data.py` crawls GitHub API with rate limit handling
- **Data Flow**: MySQL ← Django ← GitHub API → MySQL → Spark → Parquet/Models → MySQL

### Key Patterns

**Scala Data Loading Pattern**: All loaders in `utils/DatasetUtils.scala` follow `loadOrCreate` pattern - check for cached Parquet, create from MySQL if missing:
```scala
val path = s"${settings.dataDir}/${settings.today}/rawUserInfoDF.parquet"
loadOrCreateDataFrame(path, () => spark.read.jdbc(dbUrl, "app_userinfo", props))
```

**Feature Engineering Convention**: Builders categorize features into typed arrays:
```scala
val booleanColumnNames = mutable.ArrayBuffer.empty[String]
val continuousColumnNames = mutable.ArrayBuffer.empty[String]
val categoricalColumnNames = mutable.ArrayBuffer.empty[String]
val listColumnNames = mutable.ArrayBuffer.empty[String]
val textColumnNames = mutable.ArrayBuffer.empty[String]
```
This drives downstream Pipeline stages (StringIndexer, VectorAssembler, etc.)

**Two-Phase Recommendation**: Candidate generation (ALS, Content-based) produces ~100 candidates per user, then ranking model (Logistic Regression with rich features) orders them for serving.

## Critical Workflows

**Build JAR for Spark Jobs**:
```bash
mvn clean package -DskipTests  # or make build_jar
# Produces target/albedo-1.0.0-SNAPSHOT.jar
# For uber jar with dependencies: mvn clean install -DskipTests
```

**Local Development with Docker**:
```bash
make up        # Start Django, MySQL, Elasticsearch containers
make attach    # Shell into Django container
python manage.py migrate
python manage.py collect_data -t GITHUB_TOKEN -u USERNAME  # Multi-hour GitHub crawl
make run       # Start Django dev server on port 8000
```

**Spark Cluster Setup**:
```bash
make spark_start  # Standalone cluster on localhost:7077
# Or use Google Cloud Dataproc with platform=gcp flag
```

**Training Pipeline Sequence** (run in container or with spark-submit):
1. `make build_user_profile` → UserProfileBuilder → user features
2. `make build_repo_profile` → RepoProfileBuilder → repo features
3. `make train_als` → ALSRecommenderBuilder → collaborative filtering model
4. `make train_word2vec` → Word2VecCorpusBuilder → text embeddings
5. `make train_lr` → LogisticRegressionRanker → final ranking model

**Debugging Spark Jobs**: Set `RUN_WITH_INTELLIJ=true` env var to run locally with configurable memory. Check `LogisticRegressionRanker.scala:24-34` for IntelliJ-specific config block.

## Project-Specific Conventions

- **Spark Configuration**: All jobs read `spark.albedo.dataDir` and `spark.albedo.checkpointDir` from SparkConf (see `src/main/scala/ws/vinta/albedo/settings/package.scala`)
- **Date-Based Caching**: Parquet files include `${settings.today}` (yyyyMMdd format) in paths for daily versioning
- **Column Naming**: Prefix all columns with entity type - `user_*`, `repo_*` to avoid ambiguity in joins
- **Custom Recommender Contract**: Extend `recommenders/Recommender.scala`, implement `recommendForUsers(userDF)` returning DataFrame with user/item/score columns
- **Negative Sampling**: Use `NegativeBalancer` transformer for implicit feedback - generates random non-starred repos as negative examples (see `LogisticRegressionRanker.scala:257`)

## Integration Points

- **MySQL**: Hardcoded connection in `DatasetUtils.scala:11-15` (localhost:3306/albedo, user: root, password: 123)
- **Elasticsearch**: Content-based recommender queries ES More Like This API at localhost:9200 (see `ContentRecommenderBuilder.scala`)
- **GitHub API**: Rate-limited crawler with token rotation in `collect_data.py:36-48`, requires multiple tokens for production scale

## Important Files

- `Makefile`: All development commands (Docker, Spark cluster, training shortcuts)
- `src/main/scala/ws/vinta/albedo/schemas/package.scala`: Case classes define data contracts between Spark and Django
- `src/main/scala/ws/vinta/albedo/closures/UDFs.scala`: Custom Spark SQL UDFs for text cleaning and feature extraction
- `docker-compose.yml`: Local dev stack (Django, MySQL 5.7, Elasticsearch 5.6.2)
