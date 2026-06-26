import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline 

# Imputation/Scaling Tools & Models
from sklearn.impute import SimpleImputer # Using median imputation
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score

# Algorithms
from sklearn.linear_model import LogisticRegression  
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier 
from sklearn.ensemble import HistGradientBoostingClassifier

DATA_DIR   = r'C:\Users\Mahmoud\Desktop\Project - Copy\data'
TARGET_COL = 'TARGET'
ID_COL     = 'SK_ID_CURR'
SPLIT_COL  = 'is_train'


def initial_feature_filtering(df, target_col, null_threshold=0.25):
    """
    Filters columns that are entirely useless (> 75% missing).
    Returns the filtered dataframe and the list of selected feature names.
    """
    print("--- Step 1: Global Feature Filtering (Null Check) ---")
    null_ratio = df.isnull().sum() / len(df)
    # We keep columns with null ratio < threshold (i.e., less than 25% missing)
    cols_to_keep = [col for col in df.columns if col not in [target_col] and null_ratio[col] < (1 - null_threshold)]
    
    df_filtered = df.drop(columns=[c for c in df.columns if c not in cols_to_keep and c != target_col])
    print(f"Filtering complete. Features kept: {len(cols_to_keep)}")
    return df_filtered, cols_to_keep


def select_features_lgbm(df, index_cols, target_col=TARGET_COL,
                          auc_drop_tolerance=0.002, min_features=60):
    """
    Iteratively removes the bottom 10% of features by LightGBM gain
    importance, stopping once validation AUC drops more than
    `auc_drop_tolerance` below the best score seen so far.
    Returns (filtered_df, best_feature_list, best_auc).
    """
    print("\n--- Step 2: LightGBM Backward Feature Elimination ---")

    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))

    train_df = df[df[target_col].notnull()].copy()
    feature_cols = [c for c in train_df.columns if c not in index_cols + [target_col]]

    X = train_df[feature_cols]
    y = train_df[target_col]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )

    current_features = feature_cols.copy()
    best_auc = 0.0
    best_feature_set = current_features.copy()

    while len(current_features) > min_features:
        clf = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=8,
            is_unbalance=True, metric='auc', importance_type='gain',
            random_state=42, n_jobs=-1, verbose=-1
        )
        clf.fit(
            X_tr[current_features], y_tr,
            eval_set=[(X_tr[current_features], y_tr), (X_val[current_features], y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False)]
        )

        val_auc = roc_auc_score(y_val, clf.predict_proba(X_val[current_features])[:, 1])
        print(f"Features: {len(current_features):4d}  |  Val AUC: {val_auc:.5f}  |  Trees: {clf.best_iteration_}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_feature_set = current_features.copy()
        elif best_auc - val_auc > auc_drop_tolerance:
            print(f"Stopping — AUC dropped {best_auc - val_auc:.5f} below best ({best_auc:.5f})")
            break

        importances = pd.Series(clf.feature_importances_, index=current_features)
        n_to_drop = max(1, int(len(current_features) * 0.10))
        cols_to_drop = importances.sort_values(ascending=True).head(n_to_drop).index.tolist()
        current_features = [c for c in current_features if c not in cols_to_drop]

    print("\n" + "=" * 50)
    print(f"BEST RESULT: {best_auc:.5f} AUC with {len(best_feature_set)} features")
    print("=" * 50)

    final_df = df[index_cols + [target_col] + best_feature_set].copy()
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    return final_df, best_feature_set, best_auc


def clean_and_engineer_credit_data(input_parquet_path, output_df):
    """
    Comprehensive function to clean, engineer, and filter the credit dataset. 
    This step runs first and produces the cleaned feature matrix (X).
    """
    print("\n=============================================")
    print("--- STARTING FEATURE ENGINEERING PIPELINE ---")
    # Step 1 — Basic cleaning & deduplication
    df = pd.read_parquet(input_parquet_path)
    df.drop_duplicates(inplace=True)

    # STEP 2 — Binary flag encoding (assuming these features are present)
    for bin_feature in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        if bin_feature in df.columns:
            df[bin_feature], _ = pd.factorize(df[bin_feature])

    # STEP 3 — Fix sentinel values & convert DAYS_* to positive years
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, 0)
        df["YEARS_EMPLOYED"] = df["DAYS_EMPLOYED"].abs() / 365

    if "DAYS_BIRTH" in df.columns:
        df["YEARS_BIRTH"] = (df["DAYS_BIRTH"].abs() / 365).astype(int)

    # STEP 4 — Outlier capping
    if "AMT_INCOME_TOTAL" in df.columns:
        income_cap = df["AMT_INCOME_TOTAL"].quantile(0.99)
        df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].clip(upper=income_cap)

    # STEP 5 — Drop near-zero-variance columns (Threshold: 0.999)
    nzv_cols = []
    for col in df.columns:
        counts = df[col].value_counts(normalize=True, dropna=False)
        if len(counts) == 0: continue
            
        most_common_prop = counts.values[0]
        if most_common_prop > 0.999:
            nzv_cols.append(col)
    df = df.drop(columns=nzv_cols, errors='ignore')
    print(f"Dropped {len(nzv_cols)} near-zero-variance columns.")

    # STEP 6 — Drop high-missing columns (Threshold: 0.75)
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.75].index.tolist()
    df = df.drop(columns=cols_to_drop, errors='ignore')
    print(f"Dropped {len(cols_to_drop)} high-missing columns.")

    # STEP 7 — Feature engineering (Ratios and Interactions)
    try:
        df["CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"] / (df.get("AMT_GOODS_PRICE", pd.Series(1)).fillna(1) + 1e-5)
        df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df.get("AMT_INCOME_TOTAL", pd.Series(1)).replace(0, np.nan)
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df.get("AMT_INCOME_TOTAL", pd.Series(1)).replace(0, np.nan)
        df["CREDIT_TERM_MONTHS"] = df["AMT_CREDIT"] / df.get("AMT_ANNUITY", pd.Series(1)).replace(0, np.nan)
        df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"].replace(0, np.nan)
        df["EMPLOYED_TO_AGE_RATIO"] = df["YEARS_EMPLOYED"] / df["YEARS_BIRTH"].replace(0, np.nan)
        df["DAYS_EMPLOYED_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"].replace(0, np.nan)

    except KeyError as e:
        print(f"Warning during feature engineering: Missing column {e}. Skipping ratios.")
        pass # Gracefully handles if a key column is missing post-filtering/drop


    # External Source Features (Need to handle potential grouping issues dynamically)
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    present_ext_cols = [c for c in ext_cols if c in df.columns]
    if len(present_ext_cols) == 3:
        df["EXT_SOURCE_MEAN"] = df[present_ext_cols].mean(axis=1)
        df["EXT_SOURCE_PROD"] = (
            df[present_ext_cols[0]].fillna(1) * df[present_ext_cols[1]].fillna(1) * df[present_ext_cols[2]].fillna(1)
        )

    # NOTE: Categorical encoding is NOT done here anymore.
    # The OneHotEncoder inside build_linear_pipeline / build_tree_pipeline
    # handles it properly, ensuring consistent columns between train & test.

    # High Correlation Drop (Advanced Filtering - Best kept as is!)
    base_protected = ["TARGET", "SK_ID_CURR", "YEARS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_GOODS_PRICE", "EXT_SOURCE_MEAN", "EXT_SOURCE_PROD"]
    dynamic_protected = [col for col in df.columns if "RATIO" in col or "PERC" in col]
    protected_cols = set(base_protected + dynamic_protected)

    numeric_df = df.select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    raw_cols_to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.99)]
    corr_cols_to_drop = [col for col in raw_cols_to_drop if col not in protected_cols]

    df = df.drop(columns=corr_cols_to_drop, errors='ignore')
    print(f"Dropped {len(corr_cols_to_drop)} highly correlated noise columns.")


    # Memory optimisation and final save
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == "float64":
            df[col] = df[col].astype("float32")
        elif col_type == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")

    # Return the engineered DataFrame to be used for splitting
    return df 


def build_linear_pipeline(df_selected):
    """
    Builds the robust preprocessor for linear/distance/probability models. 
    Uses SimpleImputer (median) and RobustScaler on numerical data.
    """
    print("\n--- Building Linear Preprocessor (Median Impute & Scaling) ---")
    
    num_cols = df_selected.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_selected.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_pipe', Pipeline([
                ('imputer', SimpleImputer(strategy='median')), 
                ('scaler', RobustScaler())             
            ]), num_cols),
            ('cat_ohe', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ],
        remainder='passthrough'
    )
    
    return preprocessor


def build_tree_pipeline(df_selected):
    """
    Builds the minimal preprocessor for tree models. 
    Only encodes categorical features and passes through numerical data.
    """
    print("\n--- Building Tree Preprocessor (Minimal Preprocessing) ---")
    
    num_cols = df_selected.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_selected.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat_ohe', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num_pass', 'passthrough', num_cols) 
        ],
        remainder='passthrough'
    )
    
    return preprocessor


def run_full_workflow():
    global DATA_DIR 

    # --- STEP 1: Massive Feature Engineering ---
    raw_parquet_path = os.path.join(DATA_DIR, 'app_train_merged.parquet')
    df_engineered = clean_and_engineer_credit_data(raw_parquet_path, None)

    # --- STEP 1.5: Feature Selection (LightGBM Backward Elimination) ---
    # Runs on full df_engineered (uses only labeled rows internally)
    # Iteratively drops the weakest 10% of features by importance until AUC degrades
    df_engineered, selected_features, selection_auc = select_features_lgbm(
        df_engineered,
        index_cols=[ID_COL, SPLIT_COL],
        auc_drop_tolerance=0.002,
        min_features=60
    )

    # --- STEP 2: Data Splitting ---
    # 1. Isolate ONLY the data that actually has Target labels
    labeled_data = df_engineered[df_engineered[SPLIT_COL] == 1].copy()

    # 2. Separate features (X) and true targets (y)
    X_all_labeled = labeled_data.drop(columns=[TARGET_COL, SPLIT_COL, ID_COL], errors='ignore')
    y_all_labeled = labeled_data[TARGET_COL]
    
    # 3. Create a LOCAL Train/Validation split to evaluate your models
    X_train, X_val, y_train, y_val = train_test_split(
        X_all_labeled, y_all_labeled, 
        test_size=0.1, 
        random_state=42, 
        stratify=y_all_labeled 
    )

    # 4. Save the unlabelled Kaggle test set ONLY for your final submission CSV later
    X_kaggle_test = df_engineered[df_engineered[SPLIT_COL] == 0].drop(
        columns=[TARGET_COL, SPLIT_COL, ID_COL], errors='ignore'
    )
    print(f"Training on {len(selected_features)} selected features.")

    # STEP 3: Building Pipelines
    linear_pipeline = build_linear_pipeline(X_train)
    tree_pipeline = build_tree_pipeline(X_train)

    # STEP 4: Model Comparison Loop 
    results = {}

    # =======================================================================
    # A. LINEAR/DISTANCE MODELS (Optimized & Fixed)
    # =======================================================================
    print("\n=========================================================")
    
    # FIXED: Swapped LR solver to 'lbfgs' and added class_weight='balanced'
    linear_models = [
        LogisticRegression(max_iter=10000, solver='lbfgs', class_weight='balanced', random_state=42, verbose=1), 
        KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    ]

    for model in linear_models:
        model_name = type(model).__name__
        print(f"Starting {model_name} Training...")
        
        # Build pipeline steps dynamically
        pipeline_steps = [('preprocessor', linear_pipeline)]
        
        # ONLY apply undersampling to KNN (Logistic Regression uses class_weight instead)
        if model_name == 'KNeighborsClassifier':
            pipeline_steps.append(('undersample', RandomUnderSampler(random_state=42)))
            
        pipeline_steps.append(('classifier', model))
        final_pipeline = ImbPipeline(pipeline_steps)

        # Training on the 90% local training set
        final_pipeline.fit(X_train, y_train)
        
        # Predicting on the 10% local validation set
        y_pred_proba = final_pipeline.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_val, y_pred_proba)
        results[model_name] = auc
        print(f"{model_name} Validation AUC: {auc:.4f}")

    # =======================================================================
    # B. TREE MODELS (Added HistGradientBoosting & Class Balancing)
    # =======================================================================
    print("\n=========================================================")
    
    tree_models = [
        HistGradientBoostingClassifier(class_weight='balanced', random_state=42, verbose=1),
        LGBMClassifier(class_weight='balanced', random_state=42), 
        XGBClassifier(eval_metric='logloss', scale_pos_weight=11, random_state=42)  
    ]

    for model in tree_models:
        model_name = type(model).__name__
        print(f"Starting {model_name} Training...")
        
        final_pipeline = Pipeline([
            ('preprocessor', tree_pipeline),
            ('classifier', model)
        ])

        final_pipeline.fit(X_train, y_train)
        
        y_pred_proba = final_pipeline.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_val, y_pred_proba)
        results[model_name] = auc
        print(f"{model_name} Validation AUC: {auc:.4f}")

    # Final Summary 
    print("\n" + "="*60)
    print(" MODEL COMPARISON COMPLETE")
    for model, auc in results.items():
        print(f"- {model}: AUC={auc:.4f}")
        
    best_model = max(results, key=results.get)
    print(f"\n Best overall performing Model: {best_model} (AUC: {results[best_model]:.4f})")

run_full_workflow()
