from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Iris veri setini yükleyin
iris = load_iris()
X = iris.data
y = iris.target

# Veriyi eğitim ve test setlerine bölelim (%70 eğitim, %30 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Gaussian Naive Bayes modelini oluşturun ve eğitin
model = GaussianNB()
model.fit(X_train, y_train)

# Modelin performansını değerlendirin
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, target_names=iris.target_names)

# Sonuçları raporlayın
print("Doğruluk Oranı:", accuracy)
print("Karmaşıklık Matrisi:\n", conf_matrix)
print("Sınıflandırma Raporu:\n", classification_rep)
