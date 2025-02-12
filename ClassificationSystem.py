import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Preprocess as pp
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

# Funzione generica per creare dizionari di mapping
def create_mapping(dataset, column):
    unique_values = dataset[column].dropna().unique()
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    mapping['unknown'] = len(mapping)
    return mapping

# Funzione per il preprocessing del dataset
def prepDataset(dataset):
    dataset.drop(['metascore', 'no_players'], axis=1, inplace=True)
    # Creazione dei dizionari per ogni colonna categorica
    genreDict = create_mapping(dataset, 'genre')
    titleDict = create_mapping(dataset, 'title')
    yearDict = create_mapping(dataset, 'year_range')
    publisherDict = create_mapping(dataset, 'publisher')
    characteristicDict = create_mapping(dataset, 'characteristic')
    platformDict = create_mapping(dataset, 'platform')
    user_avgDict = create_mapping(dataset, 'user_avg')
    # Applicazione dei mapping
    def map_column(row, column, mapping):
        element = row[column]
        return mapping.get(element, mapping['unknown'])
    dataset['genre'] = dataset.apply(lambda row: map_column(row, 'genre', genreDict), axis=1)
    dataset['title'] = dataset.apply(lambda row: map_column(row, 'title', titleDict), axis=1)
    dataset['year_range'] = dataset.apply(lambda row: map_column(row, 'year_range', yearDict), axis=1)
    dataset['publisher'] = dataset.apply(lambda row: map_column(row, 'publisher', publisherDict), axis=1)
    dataset['characteristic'] = dataset.apply(lambda row: map_column(row, 'characteristic', characteristicDict), axis=1)
    dataset['platform'] = dataset.apply(lambda row: map_column(row, 'platform', platformDict), axis=1)
    dataset['user_avg'] = dataset.apply(lambda row: map_column(row, 'user_avg', user_avgDict), axis=1)
    # Definizione della colonna target e split del dataset
    y = dataset['genre']
    dataset.drop('genre', axis=1, inplace=True)
    X = dataset
    # Bilanciamento del dataset
    ros = RandomOverSampler(sampling_strategy="not majority")
    X_res, y_res = ros.fit_resample(X, y)
    # Split del dataset in training e test set
    xtr, xts, ytr, yts = train_test_split(X_res, y_res, test_size=0.3, random_state=0)
    # Classe per memorizzare gli elementi preprocessati
    class PrepElements:
        x_train = xtr
        y_train = ytr
        x_test = xts
        y_test = yts
        genreD = genreDict
        titleD = titleDict
        yearD = yearDict
        publisherD = publisherDict
        characteristicD = characteristicDict
        platformD = platformDict
        user_avgD = user_avgDict
        X_train_complete = X_res
        y_train_complete = y_res
    prep = PrepElements()
    return prep

def models_comparison(X, y):
    outer_kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # Outer K-Fold
    inner_kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # Inner K-Fold

    # Models to compare
    svc_model = SVC(C=1, gamma='auto', kernel='rbf', probability=True)
    rand_model = RandomForestClassifier(max_features='log2', n_estimators=100, random_state=42)
    tree_model = DecisionTreeClassifier(random_state=42)

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro', zero_division=0),
        'recall': make_scorer(recall_score, average='macro', zero_division=0),
    }

    # Combined metrics (inner + outer)
    svc_combined_metrics = {'accuracy': [], 'precision': [], 'recall': []}
    rfc_combined_metrics = {'accuracy': [], 'precision': [], 'recall': []}
    tree_combined_metrics = {'accuracy': [], 'precision': [], 'recall': []}

    # Outer K-Fold loop
    for fold, (train_idx, test_idx) in enumerate(outer_kfold.split(X, y), start=1):
        print(f"\nOuter Fold {fold}:")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Inner K-Fold: Cross-validation within the training set
        print(f"  Performing inner cross-validation for models...")

        # Inner cross-validation for SVC
        svc_inner_results = cross_validate(svc_model, X_train, y_train, cv=inner_kfold, scoring=scoring)
        svc_combined_metrics['accuracy'].append(svc_inner_results['test_accuracy'].mean())
        svc_combined_metrics['precision'].append(svc_inner_results['test_precision'].mean())
        svc_combined_metrics['recall'].append(svc_inner_results['test_recall'].mean())

        # Train final model on the full outer training data
        
        svc_model.fit(X_train, y_train)
        svc_preds = svc_model.predict(X_test)
        svc_combined_metrics['accuracy'].append(accuracy_score(y_test, svc_preds))
        svc_combined_metrics['precision'].append(precision_score(y_test, svc_preds, average='macro', zero_division=0))
        svc_combined_metrics['recall'].append(recall_score(y_test, svc_preds, average='macro', zero_division=0))

        # Inner cross-validation for Random Forest
        rfc_inner_results = cross_validate(rand_model, X_train, y_train, cv=inner_kfold, scoring=scoring)
        rfc_combined_metrics['accuracy'].append(rfc_inner_results['test_accuracy'].mean())
        rfc_combined_metrics['precision'].append(rfc_inner_results['test_precision'].mean())
        rfc_combined_metrics['recall'].append(rfc_inner_results['test_recall'].mean())

        # Train final model on the full outer training data
        rand_model.fit(X_train, y_train)
        rfc_preds = rand_model.predict(X_test)
        rfc_combined_metrics['accuracy'].append(accuracy_score(y_test, rfc_preds))
        rfc_combined_metrics['precision'].append(precision_score(y_test, rfc_preds, average='macro', zero_division=0))
        rfc_combined_metrics['recall'].append(recall_score(y_test, rfc_preds, average='macro', zero_division=0))

        # Inner cross-validation for Decision Tree
        tree_inner_results = cross_validate(tree_model, X_train, y_train, cv=inner_kfold, scoring=scoring)
        tree_combined_metrics['accuracy'].append(tree_inner_results['test_accuracy'].mean())
        tree_combined_metrics['precision'].append(tree_inner_results['test_precision'].mean())
        tree_combined_metrics['recall'].append(tree_inner_results['test_recall'].mean())

        # Train final model on the full outer training data
        tree_model.fit(X_train, y_train)
        tree_preds = tree_model.predict(X_test)
        tree_combined_metrics['accuracy'].append(accuracy_score(y_test, tree_preds))
        tree_combined_metrics['precision'].append(precision_score(y_test, tree_preds, average='macro', zero_division=0))
        tree_combined_metrics['recall'].append(recall_score(y_test, tree_preds, average='macro', zero_division=0))

    # Average the combined metrics



    models_scores_table = pd.DataFrame({
        'SVC': [
            np.mean(svc_combined_metrics['accuracy']),
            np.mean(svc_combined_metrics['precision']),
            np.mean(svc_combined_metrics['recall']),
        ],
        'Random Forest': [
            np.mean(rfc_combined_metrics['accuracy']),
            np.mean(rfc_combined_metrics['precision']),
            np.mean(rfc_combined_metrics['recall']),
        ],
        'Decision Tree': [
            np.mean(tree_combined_metrics['accuracy']),
            np.mean(tree_combined_metrics['precision']),
            np.mean(tree_combined_metrics['recall']),
        ],
    }, index=['Accuracy', 'Precision', 'Recall'])


    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)

    print("\n\nFinal Model Comparison (Combined Inner + Outer Cross-Validation Results):\n\n")
    print(models_scores_table)

    # Prepare metrics for plotting
    acc = [
        round(np.mean(svc_combined_metrics['accuracy']), 2),
        round(np.mean(rfc_combined_metrics['accuracy']), 2),
        round(np.mean(tree_combined_metrics['accuracy']), 2),
    ]
    prec = [
        round(np.mean(svc_combined_metrics['precision']), 2),
        round(np.mean(rfc_combined_metrics['precision']), 2),
        round(np.mean(tree_combined_metrics['precision']), 2),
    ]
    rec = [
        round(np.mean(svc_combined_metrics['recall']), 2),
        round(np.mean(rfc_combined_metrics['recall']), 2),
        round(np.mean(tree_combined_metrics['recall']), 2),
    ]

    return models_scores_table, prec, rec, acc

# Funzione per stampare la matrice di confusione
def plot_confusion_matrix(y_true, y_pred, model_name, genre_mapping):
    labels = [genre for genre, idx in sorted(genre_mapping.items(), key=lambda x: x[1])]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Matrice di Confusione - {model_name}')
    plt.ylabel('Reale')
    plt.xlabel('Predetto')
    plt.xticks(rotation=45)  
    plt.yticks(rotation=0)   
    plt.tight_layout()       
    plt.show()

# Funzione per addestrare un modello specifico e valutarlo
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, genre_mapping):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Stampa della matrice di confusione con etichette di classe
    plot_confusion_matrix(y_test, y_pred, model_name, genre_mapping)
    # Calcolo delle metriche
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    return accuracy, precision, recall

# Funzione per il plotting dei risultati
def plotResults(prec, rec, acc):
    labels = ['SVC', 'Random Forest', 'Decision Tree']
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, prec, width, label='Precision', align='edge')
    rects2 = ax.bar(x, rec, width, label='Recall', align='edge')
    rects3 = ax.bar(x + width, acc, width, label='Accuracy', align='edge')
    ax.set_ylabel('Scores')
    ax.set_title('Confronto tra modelli')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower left')
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    fig.tight_layout()
    plt.show()

# Funzione principale
def main(userInputGame):
    # Preprocessing del dataset
    dataset = pp.main()
    prepInfo = prepDataset(dataset)
    # Confronto tra i modelli
    models_scores_table, prec, rec, acc = models_comparison(prepInfo.X_train_complete, prepInfo.y_train_complete)
    # Addestramento e valutazione dei modelli
    svc_model = SVC(C=1.0, gamma='auto', kernel='rbf', probability=True)
    rand_model = RandomForestClassifier(max_features='sqrt', n_estimators=100, random_state=42)
    tree_model = DecisionTreeClassifier(random_state=42)
    print("\nValutazione dettagliata dei modelli:")
    # Addestramento e valutazione dei modelli
    svc_metrics = train_and_evaluate_model(svc_model, prepInfo.x_train, prepInfo.y_train, prepInfo.x_test, prepInfo.y_test, "SVC", prepInfo.genreD)
    rf_metrics = train_and_evaluate_model(rand_model, prepInfo.x_train, prepInfo.y_train, prepInfo.x_test, prepInfo.y_test, "Random Forest", prepInfo.genreD)
    tree_metrics = train_and_evaluate_model(tree_model, prepInfo.x_train, prepInfo.y_train, prepInfo.x_test, prepInfo.y_test, "Decision Tree", prepInfo.genreD)
    # Visualizzazione del grafico di confronto
    plotResults(prec, rec, acc)
    # Predizione del genere per l'input dell'utente con ciascun modello
    models = {
        "SVC": "SVC.sav",
        "Random Forest": "RandomForest.sav",
        "Decision Tree": "DecisionTree.sav"
    }
    for model_name, model_file in models.items():
        result = predictionGenre(model_file, userInputGame, prepInfo)
        print(f"Il genere del videogioco inserito secondo il modello '{model_name}' è '{result}'.")

# Funzione per la predizione del genere
def predictionGenre(filename, userInput, prepElement):
    user = userInput.copy()
    user.drop('genre', axis=1, inplace=True)
    # Mapping degli input utente
    user.at[0, 'title'] = prepElement.titleD.get(user.at[0, 'title'], prepElement.titleD['unknown'])
    user.at[0, 'year_range'] = prepElement.yearD.get(user.at[0, 'year_range'], prepElement.yearD['unknown'])
    user.at[0, 'publisher'] = prepElement.publisherD.get(user.at[0, 'publisher'], prepElement.publisherD['unknown'])
    user.at[0, 'characteristic'] = prepElement.characteristicD.get(user.at[0, 'characteristic'], prepElement.characteristicD['unknown'])
    user.at[0, 'platform'] = prepElement.platformD.get(user.at[0, 'platform'], prepElement.platformD['unknown'])
    user.at[0, 'user_avg'] = prepElement.user_avgD.get(user.at[0, 'user_avg'], prepElement.user_avgD['unknown'])
    model = joblib.load(filename)
    gen = model.predict(user)
    # Conversione del valore numerico del genere al nome corrispondente
    for genre, val in prepElement.genreD.items():
        if val == gen:
            gen = genre
            break
    return gen

if __name__ == "__main__":
    # Esempio di input utente (modificare con i dati reali)
    userInputGame = pd.DataFrame([{
        'title': 'Example Game',
        'year_range': '2015-2020',
        'publisher': 'Example Publisher',
        'characteristic': 'Fantasy',
        'platform': 'PC',
        'user_avg': '8.5'
    }])
    main(userInputGame)


import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import Preprocess as pp
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

# Funzione generica per creare dizionari di mapping
def create_mapping(dataset, column):
    unique_values = dataset[column].dropna().unique()
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    mapping['unknown'] = len(mapping)
    return mapping

# Funzione per il preprocessing del dataset
def prepDataset(dataset):
    dataset.drop(['metascore', 'no_players'], axis=1, inplace=True)
    # Creazione dei dizionari per ogni colonna categorica
    genreDict = create_mapping(dataset, 'genre')
    titleDict = create_mapping(dataset, 'title')
    yearDict = create_mapping(dataset, 'year_range')
    publisherDict = create_mapping(dataset, 'publisher')
    characteristicDict = create_mapping(dataset, 'characteristic')
    platformDict = create_mapping(dataset, 'platform')
    user_avgDict = create_mapping(dataset, 'user_avg')
    # Applicazione dei mapping
    def map_column(row, column, mapping):
        element = row[column]
        return mapping.get(element, mapping['unknown'])
    dataset['genre'] = dataset.apply(lambda row: map_column(row, 'genre', genreDict), axis=1)
    dataset['title'] = dataset.apply(lambda row: map_column(row, 'title', titleDict), axis=1)
    dataset['year_range'] = dataset.apply(lambda row: map_column(row, 'year_range', yearDict), axis=1)
    dataset['publisher'] = dataset.apply(lambda row: map_column(row, 'publisher', publisherDict), axis=1)
    dataset['characteristic'] = dataset.apply(lambda row: map_column(row, 'characteristic', characteristicDict), axis=1)
    dataset['platform'] = dataset.apply(lambda row: map_column(row, 'platform', platformDict), axis=1)
    dataset['user_avg'] = dataset.apply(lambda row: map_column(row, 'user_avg', user_avgDict), axis=1)
    # Definizione della colonna target e split del dataset
    y = dataset['genre']
    dataset.drop('genre', axis=1, inplace=True)
    X = dataset
    # Bilanciamento del dataset
    ros = RandomOverSampler(sampling_strategy="not majority")
    X_res, y_res = ros.fit_resample(X, y)
    # Split del dataset in training e test set
    xtr, xts, ytr, yts = train_test_split(X_res, y_res, test_size=0.3, random_state=0)
    # Classe per memorizzare gli elementi preprocessati
    class PrepElements:
        x_train = xtr
        y_train = ytr
        x_test = xts
        y_test = yts
        genreD = genreDict
        titleD = titleDict
        yearD = yearDict
        publisherD = publisherDict
        characteristicD = characteristicDict
        platformD = platformDict
        user_avgD = user_avgDict
        X_train_complete = X_res
        y_train_complete = y_res
    prep = PrepElements()
    return prep

# Funzione per stampare la matrice di confusione con indici numerici
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Rimuovi xticklabels e yticklabels
    plt.title(f'Matrice di Confusione - {model_name}')
    plt.ylabel('Reale')
    plt.xlabel('Predetto')
    plt.show()

# Funzione per addestrare un modello specifico e valutarlo
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Stampa della matrice di confusione
    plot_confusion_matrix(y_test, y_pred, model_name)
    # Calcolo delle metriche
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    return accuracy, precision, recall

# Funzione principale
def main(userInputGame):
    # Preprocessing del dataset
    dataset = pp.main()
    prepInfo = prepDataset(dataset)
    # Confronto tra i modelli
    models_scores_table, prec, rec, acc = models_comparison(prepInfo.X_train_complete, prepInfo.y_train_complete)
    # Addestramento e valutazione dei modelli
    svc_model = SVC(C=1.0, gamma='auto', kernel='rbf', probability=True)
    rand_model = RandomForestClassifier(max_features='sqrt', n_estimators=100, random_state=42)
    tree_model = DecisionTreeClassifier(random_state=42)
    print("\nValutazione dettagliata dei modelli:")
    svc_metrics = train_and_evaluate_model(svc_model, prepInfo.x_train, prepInfo.y_train, prepInfo.x_test, prepInfo.y_test, "SVC")
    rf_metrics = train_and_evaluate_model(rand_model, prepInfo.x_train, prepInfo.y_train, prepInfo.x_test, prepInfo.y_test, "Random Forest")
    tree_metrics = train_and_evaluate_model(tree_model, prepInfo.x_train, prepInfo.y_train, prepInfo.x_test, prepInfo.y_test, "Decision Tree")
    # Visualizzazione del grafico di confronto
    plotResults(prec, rec, acc)
    # Predizione del genere per l'input dell'utente con ciascun modello
    models = {
        "SVC": "SVC.sav",
        "Random Forest": "RandomForest.sav",
        "Decision Tree": "DecisionTree.sav"
    }
    for model_name, model_file in models.items():
        result = predictionGenre(model_file, userInputGame, prepInfo)
        print(f"Il genere del videogioco inserito secondo il modello '{model_name}' è '{result}'.")

# Funzione per la predizione del genere
def predictionGenre(filename, userInput, prepElement):
    user = userInput.copy()
    user.drop('genre', axis=1, inplace=True)
    # Mapping degli input utente
    user.at[0, 'title'] = prepElement.titleD.get(user.at[0, 'title'], prepElement.titleD['unknown'])
    user.at[0, 'year_range'] = prepElement.yearD.get(user.at[0, 'year_range'], prepElement.yearD['unknown'])
    user.at[0, 'publisher'] = prepElement.publisherD.get(user.at[0, 'publisher'], prepElement.publisherD['unknown'])
    user.at[0, 'characteristic'] = prepElement.characteristicD.get(user.at[0, 'characteristic'], prepElement.characteristicD['unknown'])
    user.at[0, 'platform'] = prepElement.platformD.get(user.at[0, 'platform'], prepElement.platformD['unknown'])
    user.at[0, 'user_avg'] = prepElement.user_avgD.get(user.at[0, 'user_avg'], prepElement.user_avgD['unknown'])
    model = joblib.load(filename)
    gen = model.predict(user)
    # Conversione del valore numerico del genere al nome corrispondente
    for genre, val in prepElement.genreD.items():
        if val == gen:
            gen = genre
            break
    return gen

if __name__ == "__main__":
    # Esempio di input utente (modificare con i dati reali)
    userInputGame = pd.DataFrame([{
        'title': 'Example Game',
        'year_range': '2015-2020',
        'publisher': 'Example Publisher',
        'characteristic': 'Fantasy',
        'platform': 'PC',
        'user_avg': '8.5'
    }])
    main(userInputGame)