# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------

import numpy as np



def hamming_distance(X, X_train):
    """
    Zwróć odległość Hamminga dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """

    A1 = np.array(X.toarray())
    A2 = np.array(X_train.toarray())

    M = (np.ones(shape=A1.shape) - A1) @ A2.transpose() + A1 @ (np.ones(shape=A2.shape) - A2).transpose()

    return M


def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
    odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.
    """

    # matrix with indices of sorted element
    indices = Dist.argsort(kind='mergesort')

    # y[indices] syntactic sugar, it creates matrix with rows sorted in order that indices are put in row of indices array

    return y[indices]





def p_y_x_knn(y, k):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
    """

    # usuwamy nadmierną liczbę sąsiadów

    reducedK = np.delete(y, range(k,y.shape[1]), axis=1)
    # następnie przechodzimy przez wszystkich sąsiadów
    matrix = []
    for i in range(reducedK.shape[0]):

        # zliczamy powtórzenia danego sąsiada w jednym rzędzie
        dict = {}
        for j in range(reducedK.shape[1]):
            if reducedK[i,j] in dict:
                dict[reducedK[i,j]] += 1
            else:
                dict[reducedK[i,j]] = 1
        # następnie obliczamy prawdopodobieństwo, że do danej klasy należy obiekt, czyli
        # dzielimy liczbę wystąpień danej klasy przez liczbę sąsiadów
        row = []
        classes = 4
        for m in range(0,classes):
            if m not in dict.keys():
                row.append(0)
            else:
                row.append(dict[m]/k)
        matrix.append(row)

    return matrix



def classification_error(p_y_x, y_true):
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """
    # błąd(jeśli klasa o największym prawdopodobieństwie, nie zgadza się z klasą w wektorze prawidłowych klas)
    # to iloraz błędnie ocenionych i wszystkich obiektów
    y_predicted = []

    for row in p_y_x:
        maxP = row[0]
        index = 0
        for i in range(len(row)):
            if row[i] >= maxP:
                maxP = row[i]
                index = i
        y_predicted.append(index)

    error = 0
    for i in range(len(y_predicted)):
        if y_predicted[i] != y_true[i]:
            error += 1

    return error/len(y_true)


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """

    k = 0
    errors = []

    sortedLabels = sort_train_labels_knn(hamming_distance(X_val,X_train), y_train)

    for i in range(len(k_values)):
        error = classification_error(p_y_x_knn(sortedLabels, k_values[i]),y_val)

        errors.append(error)

        if errors[k] > error:
            k = i


    return errors[k],k_values[k],errors



def estimate_a_priori_nb(y_train):
    """
    Wyznacz rozkład a priori p(y) każdej z klas dla obiektów ze zbioru
    treningowego.

    :param y_train: etykiety dla danych treningowych 1xN
    :return: wektor prawdopodobieństw a priori p(y) 1xM
    """
    summed_with_zero = np.bincount(y_train)
    # summed = np.delete(summed_with_zero, 0)
    return summed_with_zero / len(y_train)


def estimate_p_x_y_nb(X_train, y_train, a, b):
    """
    Wyznacz rozkład prawdopodobieństwa p(x|y) zakładając, że *x* przyjmuje
    wartości binarne i że elementy *x* są od siebie niezależne.

    :param X_train: dane treningowe NxD
    :param y_train: etykiety klas dla danych treningowych 1xN
    :param a: parametr "a" rozkładu Beta
    :param b: parametr "b" rozkładu Beta
    :return: macierz prawdopodobieństw p(x|y) dla obiektów z "X_train" MxD.
    """

    classes = 4
    matrix = []
    X_train = X_train.toarray().astype(int)

    for k in range(classes):

        row = []
        for j in range(X_train.shape[1]):
            sum = 0
            allPositions = 0
            for i in range(X_train.shape[0]):
                if X_train[i,j] == 1 and y_train[i] == k:
                    sum += 1
                    allPositions += 1
                elif y_train[i] == k:
                    allPositions+= 1
            row.append(float((sum + a -1)/(allPositions + a + b -2)))

        matrix.append(row)


    return matrix





def p_y_x_nb(p_y, p_x_1_y, X):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) dla każdej z klas z wykorzystaniem
    klasyfikatora Naiwnego Bayesa.

    :param p_y: wektor prawdopodobieństw a priori 1xM
    :param p_x_1_y: rozkład prawdopodobieństw p(x=1|y) MxD
    :param X: dane dla których beda wyznaczone prawdopodobieństwa, macierz NxD
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" NxM
    """


    X = X.toarray()

    def helper(line):
        result = X * line
        result += np.logical_not(X) - np.logical_not(X) * line
        result = np.apply_along_axis(np.prod, arr=result, axis=1)
        return result

    def helper2(line, x):
        result = np.multiply(line, x)
        return result

    def helper3(line):
        result = 1 / np.sum(line)
        return np.multiply(line, result)

    result_matrix = np.apply_along_axis(helper, axis=0, arr=np.transpose(p_x_1_y))
    result_matrix = np.apply_along_axis(helper2, axis=1, arr=result_matrix, x=p_y)
    result_matrix = np.apply_along_axis(helper3, axis=1, arr=result_matrix)


    return result_matrix




def model_selection_nb(X_train, X_val, y_train, y_val, a_values, b_values):
    """
    Wylicz bład dla różnych wartości *a* i *b*. Dokonaj selekcji modelu Naiwnego
    Byesa, wyznaczając najlepszą parę wartości *a* i *b*, tj. taką, dla której
    wartość błędu jest najniższa.
    
    :param X_train: zbiór danych treningowych N2xD
    :param X_val: zbiór danych walidacyjnych N1xD
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrów "a" do sprawdzenia
    :param b_values: lista parametrów "b" do sprawdzenia
    :return: krotka (best_error, best_a, best_b, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_a" i "best_b" to para parametrów
        "a" i "b" dla której błąd był najniższy, a "errors" - lista wartości
        błędów dla wszystkich kombinacji wartości "a" i "b" (w kolejności
        iterowania najpierw po "a_values" [pętla zewnętrzna], a następnie
        "b_values" [pętla wewnętrzna]).
    """
    errors = []
    for i in range(np.shape(a_values)[0]):
        line = []
        for j in range(np.shape(b_values)[0]):

            estimator_apriori = estimate_a_priori_nb(y_train)
            estimate_nb = estimate_p_x_y_nb(X_train, y_train, a_values[i], b_values[j])
            current = p_y_x_nb(estimator_apriori, estimate_nb, X_val)

            line.append(classification_error(current, y_val))

        errors.append(line)

    errors = np.asarray(errors)
    best_error = np.amin(errors)

    index_a, index_b = np.unravel_index(errors.argmin(), errors.shape)
    best_a = a_values[index_a]
    best_b = b_values[index_b]
    return best_error, best_a, best_b, errors