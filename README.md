# s25361_loan_approval

Temat: Kwalifikator pożyczkowy

Cel projektu: Wdrożenie modelu predykcyjnego dla wartości opisującej czy osoba o podanych parametrach kwalifikuje się do otrzymania pożyczki.

Problem do rozwiązania: Wytworzenie oraz wdrożenie modelu o zadowalającej jakości.

Wybrane źródło danych: Dane historyczne na temat udzielanych pożyczek. Dataset zawiera 45,000 rekordów.

Narzędzie do automatycznej analizy EDA: ydata_profiling - odpowiednik prandas-profiling.
Automatyczna analiza została zapisana w pliku profile_report.html.
Najbardziej warte uwagi są współczynniki korelacji pomiędzy:

1. Person_age oraz person_emp_exp
2. Person_age oraz cb_person_cred_hist_length
3. Person_education oraz cb_person_cred_hist_length
4. Loan_percent_income oraz loan_amnt

Są to 4 najwyższe współczynniki korelacji, o czym należy pamiętać przy dalszym rozwoju modelu i jego usprawnianiu.
Warto też zauważyć, iż rozkład atrybutu person_income jest bardzo wychylony ku niższym wartościom. Rozstrzał pomiędzy wartością maksymalną i minimalną jest bardzo duży, a znaczna większość wartości oscyluje w pobliżu dolnej granicy.
Wybranym narzędziem AutoML jest TPOT.
Narzędzie to przy użyciu różnych parametrów zawsze zwracało (lecz w różnej kolejności) jako najlepsze te 3 algorytmy:

1. Random forest
2. XGB
3. Gradient boosting

Każdy z nich prezentwał dokładność w okolicach 0.9 – 0.92

Zdecydowałem się przetestować pierwsze dwa z wynikami wykorzystując bibliotekę sklearn dla algorytmu random forest oraz xgboost dla algorytmu xgb:
Random forest model accuracy: 92.47%
Random forest model mae: 0.08%
XGB model accuracy: 88.11%
XGB model mae: 0.12%

W związku z tym wstępnie zdecydowałem się wybrać algorytm Random forest.

aby zobaczyć wyniki autoML należy odkomentować linijkę:
autoML(X_train, y_train)

Dane testowe są w w folderze /API/sample_data

Ścieżka do skryptu testującego /API/client.py

Wyniki testów znajdują się w folderze /API/sample_data/sample_data_res

odpalenie apache airflow (po sklonowaniu repo):
docker compose up --build

odpalenie api:
docker pull jaroslawgawrych/s25361_loan_approval_api:latest
docker run -d -p 5000:5000 jaroslawgawrych/s25361_loan_approval_api:latest
