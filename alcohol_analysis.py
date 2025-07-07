# ==============================================================================
# PROBLÉMÁS ALKOHOLFOGYASZTÁS ELŐREJELZÉSE BRFSS 2015 ADATOKKAL
# ==============================================================================
#
# CÉL:
# A Behavioral Risk Factor Surveillance System (BRFSS) 2015-ös adathalmazán
# gépi tanulási modellekkel azonosítani a problémás alkoholfogyasztás
# kockázati tényezőit és előrejelezni a magas kockázatú egyéneket.
#
# SZERZŐ: [A Te Neved]
# DÁTUM: [Mai Dátum]
#
# ==============================================================================

# --- 1. CSOMAGOK IMPORTÁLÁSA ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# --- 2. KONFIGURÁCIÓ ÉS GLOBÁLIS VÁLTOZÓK ---

# Adatforrás (feltételezi, hogy a szkripttel egy szinten van egy 'data' mappa)
FILE_PATH = 'data/2015.csv'

# Modellhez használt jellemzők
FEATURE_COLUMNS = [
    '_AGEG5YR', 'SEX', '_RACE', 'INCOME2', 'EDUCA', 'GENHLTH', 'PHYSHLTH',
    'MENTHLTH', 'HLTHPLN1', '_SMOKER3', '_TOTINDA', 'ALCDAY5', 'DRNKANY5',
    'DIABETE3', '_MICHD', 'CVDSTRK3', '_BMI5', '_FRUTSUM', '_VEGESUM'
]

# A célváltozó definíciójához szükséges oszlopok
TARGET_DEFINITION_COLS = ['_DRNKWEK', 'MAXDRNKS', '_RFBING5', 'SEX']

# Célváltozó neve
TARGET_COLUMN = 'alcohol_problem'


# --- 3. ADATELŐKÉSZÍTŐ FÜGGVÉNYEK ---

def clean_brfss_data(df, feature_cols, target_def_cols):
    """BRFSS specifikus kódok cseréje (pl. 9 -> NaN) és alapvető adattisztítás."""
    df_clean = df.copy()
    replace_dict = {
        7: np.nan, 9: np.nan, 77: np.nan, 99: np.nan, 777: np.nan,
        999: np.nan, 888: 0, 99999: np.nan
    }
    cols_to_clean = list(set(feature_cols + target_def_cols))
    for col in cols_to_clean:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].replace(replace_dict)
    if 'DRNKANY5' in df_clean.columns:
        df_clean['DRNKANY5'] = df_clean['DRNKANY5'].replace({2: 0})
    return df_clean

def create_alcohol_target(df):
    """Célváltozó létrehozása: Problémás alkoholfogyasztás a CDC ajánlásai alapján."""
    problem_score = pd.Series(0, index=df.index)
    heavy_men = (df['SEX'] == 1) & (df['_DRNKWEK'] > 1400)
    heavy_women = (df['SEX'] == 2) & (df['_DRNKWEK'] > 700)
    problem_score += (heavy_men | heavy_women).astype(int)
    problem_score += (df['_RFBING5'] == 2).astype(int)
    high_max_men = (df['SEX'] == 1) & (df['MAXDRNKS'] > 6)
    high_max_women = (df['SEX'] == 2) & (df['MAXDRNKS'] > 5)
    problem_score += (high_max_men | high_max_women).astype(int)
    return (problem_score >= 2).astype(int)


# --- 4. VIZUALIZÁCIÓS ÉS ELEMZŐ FÜGGVÉNYEK ---

def plot_model_results(results):
    """Modelleredmények vizualizálása: ROC görbék."""
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(result['y_test'], result['probabilities'])
        plt.plot(fpr, tpr, label=f"{name.replace('_', ' ').title()} (AUC = {result['auc']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Véletlen tipp (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Modellek ROC Görbéje')
    plt.legend()
    plt.grid(True)
    plt.savefig('roc_curves.png')
    plt.show()

def plot_feature_importance(model, feature_names):
    """Jellemző fontosság vizualizálása a legjobb modellhez."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        df_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
        df_imp = df_imp.sort_values('importance', ascending=False).head(15)
        plt.figure(figsize=(10, 8))
        sns.barplot(data=df_imp, x='importance', y='feature', palette='viridis')
        plt.title('Top 15 Legfontosabb Jellemző (LightGBM)')
        plt.xlabel('Fontosság'); plt.ylabel('Jellemző')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()

def create_risk_profiles(df, target_col):
    """Kockázati profilok létrehozása a pontosabb mentális egészség mutatóval."""
    high_risk = df[df[target_col] == 1].copy()
    low_risk = df[df[target_col] == 0].copy()
    
    # MENTHLTH: a 88-as kód a '0 nap'-ot jelenti, ezt is kezelni kell.
    for data in [high_risk, low_risk]:
        if 'MENTHLTH' in data.columns:
            data['MENTHLTH'] = data['MENTHLTH'].replace({88:0})

    age_map = {
        1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
        7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74",
        12: "75-79", 13: "80+"
    }

    profiles = {
        'Magas Kockázat': {
            'Medián Korcsoport': age_map.get(high_risk['_AGEG5YR'].median(), 'N/A'),
            'Férfi Arány': (high_risk['SEX'] == 1).mean(),
            'Jelenlegi Dohányosok Aránya': (high_risk['_SMOKER3'].isin([1, 2])).mean(),
            'Medián Rossz Mentális Napok Száma': high_risk['MENTHLTH'].median(),
        },
        'Alacsony Kockázat': {
            'Medián Korcsoport': age_map.get(low_risk['_AGEG5YR'].median(), 'N/A'),
            'Férfi Arány': (low_risk['SEX'] == 1).mean(),
            'Jelenlegi Dohányosok Aránya': (low_risk['_SMOKER3'].isin([1, 2])).mean(),
            'Medián Rossz Mentális Napok Száma': low_risk['MENTHLTH'].median(),
        }
    }
    return profiles

# --- 5. GÉPI TANULÁSI MŰVELETEK ---

class AlcoholPredictionPipeline:
    """Gépi tanulási folyamat becsomagolása egy osztályba."""
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
            'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
            'LightGBM': lgb.LGBMClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        }
        self.scaler = StandardScaler()
        self.results = {}

    def prepare_data(self, df, target_col, feature_cols):
        """Adatok előkészítése: hiányzó értékek pótlása és szétválasztás."""
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        X = X.apply(lambda x: x.fillna(x.median()), axis=0)
        return X, y

    def train_and_evaluate(self, X, y):
        """Modellek tanítása és kiértékelése."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        for name, model in self.models.items():
            print(f"\n--- {name} tanítása ---")
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            y_pred = (y_pred_proba >= 0.5).astype(int)
            auc = roc_auc_score(y_test, y_pred_proba)
            self.results[name] = {'model': model, 'auc': auc, 'probabilities': y_pred_proba, 'y_test': y_test}
            print(f"AUC: {auc:.4f}")
            print("Klasszifikációs jelentés:\n", classification_report(y_test, y_pred, target_names=['Normál', 'Problémás']))
        
        best_name = max(self.results, key=lambda name: self.results[name]['auc'])
        best_model = self.results[best_name]['model']
        print(f"\nA legjobb modell: {best_name} (AUC: {self.results[best_name]['auc']:.4f})")
        
        plot_feature_importance(best_model, X.columns)
        plot_model_results(self.results)

# --- 6. FŐPROGRAM ---

def main():
    """A teljes adatfeldolgozási és modellezési folyamat futtatása."""
    print(">>> 1. ADATOK BETÖLTÉSE ÉS ELŐKÉSZÍTÉSE")
    try:
        df = pd.read_csv(FILE_PATH, low_memory=False)
        print(f"Adatok betöltve: {df.shape[0]} sor, {df.shape[1]} oszlop.")
    except FileNotFoundError:
        print(f"HIBA: A(z) '{FILE_PATH}' fájl nem található.")
        print("Kérjük, kövesse a README.md-ben leírtakat az adatok letöltéséhez és elhelyezéséhez.")
        return

    df_clean = clean_brfss_data(df, FEATURE_COLUMNS, TARGET_DEFINITION_COLS)
    df_clean[TARGET_COLUMN] = create_alcohol_target(df_clean)
    problem_rate = df_clean[TARGET_COLUMN].mean()
    print(f"Problémás alkoholfogyasztás aránya: {problem_rate:.2%}")
    if problem_rate == 0: return

    print("\n>>> 2. MODELLEZÉS")
    pipeline = AlcoholPredictionPipeline()
    X, y = pipeline.prepare_data(df_clean, TARGET_COLUMN, FEATURE_COLUMNS)
    pipeline.train_and_evaluate(X, y)

    print("\n>>> 3. KOCKÁZATI PROFILOK ELEMZÉSE")
    profiles = create_risk_profiles(df_clean, TARGET_COLUMN)
    for risk_type, profile_data in profiles.items():
        print(f"\n--- {risk_type} ---")
        for key, value in profile_data.items():
            if isinstance(value, float) and (0 < value < 1):
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value}")

    print("\n>>> 4. EREDMÉNYEK MENTÉSE")
    df_clean.to_csv('brfss_2015_alcohol_analysis.csv', index=False)
    print("Elemzésre kész adathalmaz mentve: brfss_2015_alcohol_analysis.csv")
    print("Grafikonok mentve: roc_curves.png, feature_importance.png")
    print("\n=== FOLYAMAT SIKERESEN BEFEJEZVE ===")

if __name__ == "__main__":
    main()