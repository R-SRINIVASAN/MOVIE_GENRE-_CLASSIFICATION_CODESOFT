import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tabulate import tabulate
from colorama import Fore, Style, init

#   colorama
init(autoreset=True)

#  Loading dataset
csv_path = r"C:\Users\wishv\OneDrive\Documents\movies.csv"
df = pd.read_csv(csv_path)

#  Drop missing values
df.dropna(subset=['Plot', 'Genre'], inplace=True)

#  Features and labels
X = df['Plot']
y = df['Genre']

#  TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = tfidf.fit_transform(X)

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

#  Train model
model = MultinomialNB()
model.fit(X_train, y_train)

#  Predict on test set
y_pred = model.predict(X_test)

#  Show accuracy
print(f"\n{Fore.GREEN}{Style.BRIGHT}✅ Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

#  Build enhanced classification report
labels = model.classes_
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=labels)

table_data = []
for i in range(len(labels)):
    table_data.append([
        Fore.YELLOW + Style.BRIGHT + labels[i] + Style.RESET_ALL,
        f"{precision[i]:.2f}",
        f"{recall[i]:.2f}",
        f"{f1[i]:.2f}",
        f"{support[i]}"
    ])

headers = [
    Fore.CYAN + Style.BRIGHT + "Genre" + Style.RESET_ALL,
    Fore.CYAN + "Precision",
    "Recall",
    "F1-Score",
    "Support"
]

#  Display classification report in table
print(f"\n{Fore.MAGENTA}{Style.BRIGHT}📊 Classification Report:\n")
print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

#  Repeated User Input for Genre Prediction
print(f"\n{Fore.CYAN}{Style.BRIGHT}🎬 Genre Prediction Tool")
print(f"{Fore.YELLOW}Type 'exit' to quit the prediction tool.\n")

while True:
    user_plot = input(Fore.BLUE + "Enter a movie plot: ").strip()
    if user_plot.lower() == "exit":
        print(Fore.LIGHTRED_EX + "👋 Exiting Genre Prediction Tool. Thank you!")
        break

    if not user_plot:
        print(Fore.RED + "⚠️ Please enter a valid plot.")
        continue

    #  Predict genre
    user_plot_tfidf = tfidf.transform([user_plot])
    predicted_genre = model.predict(user_plot_tfidf)[0]
    print(f"{Fore.GREEN}{Style.BRIGHT}🔮 Predicted Genre: {predicted_genre}\n")



#Test data
#A detective investigates a series of mysterious disappearances in a small town.
#A young archaeologist embarks on a thrilling quest across the Egyptian desert to uncover a hidden tomb said to hold the secrets of the gods.
#After losing her husband in a tragic accident, a woman struggles to rebuild her life while raising her autistic son.